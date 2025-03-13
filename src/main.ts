import { WGS84_ELLIPSOID, TilesRenderer } from '3d-tiles-renderer';
import {
  TilesFadePlugin,
  TileCompressionPlugin,
  GLTFExtensionsPlugin,
  CesiumIonAuthPlugin,
} from '3d-tiles-renderer/plugins';
import Papa from 'papaparse';
import {
  Scene,
  WebGLRenderer,
  PerspectiveCamera,
  MathUtils,
  Raycaster,
  Vector3,
  Vector2,
} from 'three';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js';

import { MESSAGE_TYPES, PATH_STATUS } from './constants.js';

const MIN_ABOVE_GROUND_DISTANCE = 20;
const MAX_CONSECUTIVE_MISSING_INTERSECTIONS = 10;
const MESSAGE_DELAY_IN_MS = 5000;
const PATH_WAIT_DELAY_IN_MS = 100;

const RENDERER_WIDTH = 1024;
const RENDERER_HEIGHT = 1024;

// Global flags
let tilesLoading = false;
let consecutiveMissingIntersections = 0;
let currentPathPending = false;

type Point = {
  altitude: number;
  distance_from_ground: number;
  heading: number;
  index_in_path: number;
  lat: number;
  lng: number;
  roll: number;
  tilt: number;
};

type WGS84Point = {
  height: number;
  lat: number;
  lon: number;
};

function isArrayOfString(value: any) {
  if (!Array.isArray(value)) {
    return false;
  }
  return !value.some((item) => typeof item !== 'string');
}

function setHeading(camera: PerspectiveCamera, degrees: number) {
  camera.rotation.order = 'YXZ';
  camera.rotation.y = -(degrees + 90) * MathUtils.DEG2RAD;
  camera.updateMatrixWorld();
}

function onWindowResize(camera: PerspectiveCamera, renderer: WebGLRenderer) {
  camera.aspect = 1; // Aspect ratio
  renderer.setSize(RENDERER_WIDTH, RENDERER_HEIGHT);
  camera.updateProjectionMatrix();
  renderer.setPixelRatio(2); // Use fixed pixel ratio for consistent output, higher nubmers give better quality, but can be pretty slow if gpu is shitty, or using cpu/igpu.
}

function getLookAtPoint(tiles: TilesRenderer, camera: PerspectiveCamera, raycaster: Raycaster) {
  raycaster.setFromCamera({ x: 0, y: 0 } as Vector2, camera);
  const intersects = raycaster.intersectObject(tiles.group, true);

  // TODO: what if there are multiple intersections?
  if (intersects.length > 0) {
    const point = intersects[0].point;
    // Convert world position to local position in tiles coordinate system
    const mat = tiles.group.matrixWorld.clone().invert();
    const vec = point.clone().applyMatrix4(mat);

    const res = {} as WGS84Point;
    WGS84_ELLIPSOID.getPositionToCartographic(vec, res);

    return {
      lat: res.lat * MathUtils.RAD2DEG,
      lng: res.lon * MathUtils.RAD2DEG,
      elevation: res.height,
      distance: intersects[0].distance, // Add distance to return value
    };
  }

  return null;
}

function getGroundDistance(tiles: TilesRenderer, camera: PerspectiveCamera) {
  // Create a new raycaster pointing straight down from the camera
  const downRaycaster = new Raycaster();

  // Set the raycaster origin to the camera position
  downRaycaster.set(camera.position.clone(), new Vector3(0, -1, 0));

  // Intersect with the tiles
  const intersects = downRaycaster.intersectObject(tiles.group, true);

  // TODO: what if there are multiple intersections?
  if (intersects.length > 0) {
    const point = intersects[0].point;

    // Convert world position to local position in tiles coordinate system
    const mat = tiles.group.matrixWorld.clone().invert();
    const vec = point.clone().applyMatrix4(mat);

    const res = {} as WGS84Point;
    WGS84_ELLIPSOID.getPositionToCartographic(vec, res);

    return {
      lat: res.lat * MathUtils.RAD2DEG,
      lng: res.lon * MathUtils.RAD2DEG,
      elevation: res.height,
      distance: intersects[0].distance, // Distance from camera to ground
    };
  }

  return null;
}

function updatePathStatus(
  pathNumber: number,
  status: string,
  url: string,
  reason: string | null = null
) {
  console.log(
    `${MESSAGE_TYPES.PATH_STATUS}_${JSON.stringify({
      pathNumber,
      status,
      reason,
      url,
      timestamp: new Date().toISOString(),
    })}`
  );
}

function reinstantiateTiles(
  tiles: TilesRenderer,
  camera: PerspectiveCamera,
  renderer: WebGLRenderer,
  scene: Scene
) {
  // Force higher detail for more distant tiles
  tiles.errorTarget = 0.1;
  tiles.errorThreshold = Infinity;
  tiles.maxDepth = Infinity;

  // Load tiles even if they are not in the camera frustum
  tiles.autoDisableRendererCulling = false;

  tiles.registerPlugin(
    new CesiumIonAuthPlugin({
      apiToken: import.meta.env.VITE_ION_KEY,
      assetId: '2275207',
      autoRefreshToken: true,
    })
  );
  tiles.registerPlugin(new TileCompressionPlugin());
  tiles.registerPlugin(new TilesFadePlugin());
  tiles.registerPlugin(
    new GLTFExtensionsPlugin({
      dracoLoader: new DRACOLoader().setDecoderPath(
        'https://unpkg.com/three@0.153.0/examples/jsm/libs/draco/gltf/'
      ),
    })
  );

  scene.add(tiles.group);

  tiles.setResolutionFromRenderer(camera, renderer);
  tiles.optimizeRaycast = true;
  tiles.setCamera(camera);
}

async function loadPath(currentPathNumber: number, csvURL: string) {
  try {
    // First check if the file exists without fetching its contents
    const checkResponse = await fetch(csvURL, { method: 'HEAD' });

    // Invalid responses should be ignored
    // and skipped to the next path.
    if (!checkResponse.ok) {
      console.log(`Invalid response was received for Path #${currentPathNumber} (${csvURL})`);
      updatePathStatus(currentPathNumber, PATH_STATUS.DISCARDED, csvURL, 'no valid data');
      return [];
    }

    // If file exists, proceed with loading
    const response = await fetch(csvURL);
    const csvText = await response.text();

    // TODO: maybe further validation that it's a valid CSV file
    const results = Papa.parse(csvText, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
    });

    // Skip to the next path if there are no points
    if (results.errors.length > 0) {
      console.log(`Path #${currentPathNumber} has no valid data`);
      updatePathStatus(
        currentPathNumber,
        PATH_STATUS.DISCARDED,
        csvURL,
        results.errors.map((e) => e.message).join(', ')
      );
      return [];
    }

    // TODO: ensure that newPathData is Point[]
    const newPathData = results.data as Point[];

    console.log(`Loaded Path #${currentPathNumber} with ${newPathData.length} entries.`);

    // Skip to the next path if there are no points
    if (newPathData.length <= 0) {
      console.log(`Path ${currentPathNumber} has no points`);
      updatePathStatus(currentPathNumber, PATH_STATUS.DISCARDED, csvURL, 'no points');
      return [];
    }

    return newPathData;
  } catch (e: any) {
    console.error(`Error while loading Path #${currentPathNumber}:`, e);
    updatePathStatus(currentPathNumber, PATH_STATUS.DISCARDED, csvURL, `error: ${e.message}`);
    return [];
  }
}

async function animate(
  tiles: TilesRenderer,
  camera: PerspectiveCamera,
  raycaster: Raycaster,
  renderer: WebGLRenderer,
  scene: Scene,
  totalPathsCount: number,
  currentPathNumber: number,
  csvURL: string,
  currentPathData: Point[],
  currentFrameIndex: number
) {
  // Once the path has completed,
  // try to load the next path,
  // exiting if there are no more paths to be processed.
  if (currentFrameIndex >= currentPathData.length) {
    console.log(`Path ${currentPathNumber} complete.`);
    updatePathStatus(currentPathNumber, PATH_STATUS.COMPLETED, csvURL);

    if (currentPathNumber >= totalPathsCount) {
      console.log('All paths complete');
      console.log(MESSAGE_TYPES.PROCESSING_COMPLETE);
    }

    currentPathPending = false;

    return;
  }

  // Re-position camera for current point
  const currentPoint = currentPathData[currentFrameIndex];
  // @ts-ignore
  tiles.setLatLonToYUp(currentPoint.lat * MathUtils.DEG2RAD, currentPoint.lng * MathUtils.DEG2RAD);
  camera.position.set(0, currentPoint.altitude, 0);
  camera.rotation.order = 'YXZ';
  camera.rotation.set(
    -(90 - currentPoint.tilt) * MathUtils.DEG2RAD,
    -(currentPoint.heading + 90) * MathUtils.DEG2RAD,
    currentPoint.roll * MathUtils.DEG2RAD
  );
  tiles.setResolutionFromRenderer(camera, renderer);
  tiles.setCamera(camera);
  camera.updateMatrixWorld();
  tiles.update();

  // Render the scene
  renderer.render(scene, camera);

  // If tiles aren't loaded yet, retry the frame
  // TODO: this opens it up to infinite loops!
  if (tiles.loadProgress < 1 || !tilesLoading) {
    console.warn(
      `Tiles not loaded yet, retrying frame #${
        currentFrameIndex + 1
      } for Path #${currentPathNumber}...`
    );
    requestAnimationFrame(() =>
      animate(
        tiles,
        camera,
        raycaster,
        renderer,
        scene,
        totalPathsCount,
        currentPathNumber,
        csvURL,
        currentPathData,
        currentFrameIndex
      )
    );
    return;
  }

  // Get the point that the camera is looking at
  const lookAtPoint = getLookAtPoint(tiles, camera, raycaster);

  // If there's a missing intersection,
  // skip the current frame and go to the next one,
  // unless we've already exhausted the acceptable number of missing intersections for the path,
  // in which case, we skip to the next path.
  if (!lookAtPoint) {
    consecutiveMissingIntersections++;
    console.log(`No intersection found (${consecutiveMissingIntersections} in a row)`);

    // Check if we need to skip this path due to distance issues
    if (consecutiveMissingIntersections >= MAX_CONSECUTIVE_MISSING_INTERSECTIONS) {
      console.log(
        `${MAX_CONSECUTIVE_MISSING_INTERSECTIONS} consecutive missing intersections for path ${currentPathNumber}, skipping to next path`
      );
      updatePathStatus(
        currentPathNumber,
        PATH_STATUS.DISCARDED,
        csvURL,
        `${MAX_CONSECUTIVE_MISSING_INTERSECTIONS} consecutive missing intersections`
      );

      if (currentPathNumber >= totalPathsCount) {
        console.log('All paths complete');
        console.log(MESSAGE_TYPES.PROCESSING_COMPLETE);
      }

      currentPathPending = false;

      return;
    }

    // If we haven't hit the threshold, continue to next frame
    requestAnimationFrame(() =>
      animate(
        tiles,
        camera,
        raycaster,
        renderer,
        scene,
        totalPathsCount,
        currentPathNumber,
        csvURL,
        currentPathData,
        currentFrameIndex + 1
      )
    );
    return;
  }

  // Discard the entire path if the intersection is too close to the ground
  if (lookAtPoint.distance < MIN_ABOVE_GROUND_DISTANCE) {
    console.log(
      `Intersection too close (${lookAtPoint.distance.toFixed(
        2
      )}m) for path ${currentPathNumber}, skipping to next path`
    );
    updatePathStatus(
      currentPathNumber,
      PATH_STATUS.DISCARDED,
      csvURL,
      `intersection too close (${lookAtPoint.distance.toFixed(2)}m)`
    );

    if (currentPathNumber >= totalPathsCount) {
      console.log('All paths complete');
      console.log(MESSAGE_TYPES.PROCESSING_COMPLETE);
    }

    currentPathPending = false;

    return;
  }

  // Reset counter when we find an intersection
  consecutiveMissingIntersections = 0;

  const metadata = {
    pathNumber: currentPathNumber,
    frameIndex: currentFrameIndex,
    latitude: currentPoint.lat,
    longitude: currentPoint.lng,
    altitude: currentPoint.altitude,
    heading: currentPoint.heading,
    tilt: currentPoint.tilt,
    roll: currentPoint.roll,
    width: RENDERER_WIDTH,
    height: RENDERER_HEIGHT,
    timestamp: new Date().toISOString(),
    lookAtPoint: lookAtPoint,
    groundDistance: getGroundDistance(tiles, camera),
  };

  console.log(`${MESSAGE_TYPES.FRAME_READY}_${JSON.stringify(metadata)}`);

  // Wait for Puppeteer to capture the frame,
  // unless we hit a timeout,
  // in which case we continue to the next frame.
  try {
    await new Promise((resolve) => {
      const messageHandler = (event: MessageEvent) => {
        if (event.data === MESSAGE_TYPES.FRAME_CAPTURED) {
          window.removeEventListener('message', messageHandler);
          resolve(undefined);
        }
      };

      // Send message to Puppeteer
      window.addEventListener('message', messageHandler);
      window.postMessage(MESSAGE_TYPES.FRAME_READY, '*');

      // Make sure to resolve the Promise,
      // even if the message is never received.
      setTimeout(() => {
        window.removeEventListener('message', messageHandler);
        console.warn('Frame capture confirmation timed out');
        resolve(undefined);
      }, MESSAGE_DELAY_IN_MS);
    });
  } catch (error) {
    console.error('Error during frame capture:', error);
  } finally {
    requestAnimationFrame(() =>
      animate(
        tiles,
        camera,
        raycaster,
        renderer,
        scene,
        totalPathsCount,
        currentPathNumber,
        csvURL,
        currentPathData,
        currentFrameIndex + 1
      )
    );
  }
}

async function main() {
  const scene = new Scene();
  const raycaster = new Raycaster();

  const renderer = new WebGLRenderer({ antialias: true });
  renderer.setClearColor(0x87ceeb);
  document.body.appendChild(renderer.domElement);

  // TODO: is the fov fixed? is the near plane fixed? is the far plane fixed?
  const camera = new PerspectiveCamera(35, window.innerWidth / window.innerHeight, 1, 16000000);
  const tiles = new TilesRenderer();

  reinstantiateTiles(tiles, camera, renderer, scene);

  // Fetch the CSV URLs from the query string
  const urlParams = new URLSearchParams(window.location.search);
  const rawCsvUrls = urlParams.get('csvUrls') || (import.meta.env.VITE_DEBUG_CSV_URLS as string);
  const csvUrls = rawCsvUrls.split(',');

  if (!isArrayOfString(csvUrls) || csvUrls.length === 0) {
    console.error('No CSV URLs provided');
    return;
  }

  onWindowResize(camera, renderer);
  window.addEventListener('resize', () => onWindowResize(camera, renderer), false);
  tiles.addEventListener('tiles-load-start', () => {
    console.log('Tiles loaded');
    tilesLoading = true;
  });

  for (let currentPathNumber = 1; currentPathNumber <= csvUrls.length; currentPathNumber++) {
    // Reset all state for new path
    tilesLoading = false;
    consecutiveMissingIntersections = 0;
    currentPathPending = true;

    const currentCsvUrl = csvUrls[currentPathNumber - 1];
    const currentPathData = await loadPath(currentPathNumber, currentCsvUrl);

    if (currentPathData.length > 0) {
      // Move camera to the first point
      const [firstPoint] = currentPathData;
      // TODO: isn't this +100 throwing off the distance calculations?
      camera.position.set(0, firstPoint.altitude + 100, 0);
      // @ts-ignore
      tiles.setLatLonToYUp(firstPoint.lat * MathUtils.DEG2RAD, firstPoint.lng * MathUtils.DEG2RAD);

      // TODO: validate if rotation is supposed to be set on each new path???? or if it should be on (first path, first frame)
      camera.rotation.set(
        -(90 - firstPoint.tilt) * MathUtils.DEG2RAD, // X (tilt)
        -(firstPoint.heading + 90) * MathUtils.DEG2RAD, // Y (heading)
        firstPoint.roll * MathUtils.DEG2RAD // Z (roll)
      );
      setHeading(camera, firstPoint.heading);

      animate(
        tiles,
        camera,
        raycaster,
        renderer,
        scene,
        csvUrls.length,
        currentPathNumber,
        currentCsvUrl,
        currentPathData,
        0
      );

      // Wait for the current path to complete before moving to the next one
      while (currentPathPending) {
        await new Promise((_resolve) => setTimeout(_resolve, PATH_WAIT_DELAY_IN_MS));
      }
    }
  }
}

main();
