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
  WebGLRenderTarget,
  ShaderMaterial,
  PlaneGeometry,
  Mesh,
  OrthographicCamera,
  DepthTexture,
  NearestFilter,
  FloatType,
  TypedArray,
} from 'three';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js';

import { MESSAGE_TYPES, PATH_STATUS } from './constants.js';

const MIN_ABOVE_GROUND_DISTANCE = 40;
const MAX_ABOVE_GROUND_DISTANCE = 20_000;
const MAX_CONSECUTIVE_MISSING_INTERSECTIONS = 10;
const MAX_NUMBER_OF_FRAME_RETRIES = 5000;
const MESSAGE_DELAY_IN_MS = 5000;
const PATH_WAIT_DELAY_IN_MS = 100;
const SKIP_PATH_AFTER_EXCESSIVE_MISSING_INTERSECTIONS = true;
const SKIP_PATH_AFTER_COLLISION_WITH_GROUND = true;
const SKIP_PATH_AFTER_UNREALISTIC_HEIGHT = true;

const RENDERER_WIDTH = 1024;
const RENDERER_HEIGHT = 768;

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

function onWindowResize(
  camera: PerspectiveCamera,
  renderer: WebGLRenderer,
  depthTarget: WebGLRenderTarget
) {
  camera.aspect = 1;
  renderer.setSize(RENDERER_WIDTH, RENDERER_HEIGHT);
  depthTarget.setSize(RENDERER_WIDTH, RENDERER_HEIGHT);

  // TODO: these values are fixed when shaderMaterial is defined, keeping for bookeepping's sake
  // shaderMaterial.uniforms.cameraNear.value = camera.near;
  // shaderMaterial.uniforms.cameraFar.value = camera.far;
  // shaderMaterial.uniforms.texelSize.value.set(1.0 / RENDERER_WIDTH, 1.0 / RENDERER_HEIGHT);
  // shaderMaterial.uniforms.minDepth.value = minDepth;
  // shaderMaterial.uniforms.maxDepth.value = maxDepth;

  camera.updateProjectionMatrix();

  // Use fixed pixel ratio for consistent output.
  // Higher nubmers give better quality,
  // but can be pretty slow if GPU is shitty (or using CPU/iGPU).
  renderer.setPixelRatio(2);
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

function getPathNumberFromCsvUrl(csvUrl: string) {
  // Example URL: https://storage.googleapis.com/tera-public/terrarium-input/03-13-25/vm1/path_1.csv
  const portions = csvUrl.split('/');
  const currentPathNumberString = portions[portions.length - 1]
    .replace('path_', '')
    .replace('.csv', '');
  return Number(currentPathNumberString);
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
  scene: Scene,
  token: string
) {
  // Force higher detail for more distant tiles
  tiles.errorTarget = 0.1;
  tiles.errorThreshold = Infinity;
  tiles.maxDepth = Infinity;

  // Load tiles even if they are not in the camera frustum
  tiles.autoDisableRendererCulling = false;

  tiles.registerPlugin(
    new CesiumIonAuthPlugin({
      apiToken: token,
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

function setupDepthRendering(cameraNear: number, cameraFar: number, useLogarithmicDepth: boolean) {
  // Create render target with depth texture
  const depthTarget = new WebGLRenderTarget(RENDERER_WIDTH, RENDERER_HEIGHT);
  depthTarget.texture.minFilter = NearestFilter;
  depthTarget.texture.magFilter = NearestFilter;
  depthTarget.depthTexture = new DepthTexture(RENDERER_WIDTH, RENDERER_HEIGHT);
  depthTarget.depthTexture.type = FloatType;

  // Setup post-processing for depth visualization
  const depthCamera = new OrthographicCamera(-1, 1, 1, -1, 0, 1);

  const shaderMaterial = new ShaderMaterial({
    vertexShader: `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      #include <packing>
      varying vec2 vUv;
      uniform sampler2D tDepth;
      uniform float cameraNear;
      uniform float cameraFar;
      uniform float minDepth;
      uniform float maxDepth;
      uniform bool u_useLogarithmicDepth;
      uniform vec2 texelSize;

      float readDepth(sampler2D depthSampler, vec2 coord) {
        float fragCoordZ = texture2D(depthSampler, coord).x;
        float viewZ = perspectiveDepthToViewZ(fragCoordZ, cameraNear, cameraFar);
        float linearDepth = -viewZ;  // Convert to positive distance from camera

        // Ensure inputs are positive and avoid issues
        float clampedMinDepth = max(minDepth, 0.00001);
        float clampedMaxDepth = max(maxDepth, clampedMinDepth + 0.00001); // Ensure max > min
        float clampedLinearDepth = clamp(linearDepth, clampedMinDepth, clampedMaxDepth); // Clamp depth to the custom range
 
        float normalizedDepth;
        if (u_useLogarithmicDepth) {
          // Logarithmic depth normalization using custom min/max
          normalizedDepth = (log(clampedLinearDepth) - log(clampedMinDepth)) / (log(clampedMaxDepth) - log(clampedMinDepth));
        } else {
          // Linear depth normalization using custom min/max
          normalizedDepth = (clampedLinearDepth - clampedMinDepth) / (clampedMaxDepth - clampedMinDepth);
        }
 
        // Clamp result to [0, 1] range
        return clamp(normalizedDepth, 0.0, 1.0);
      }

      void main() {
        float centerDepth = readDepth(tDepth, vUv);
        float finalDepth = centerDepth;

        // Simple gap filling: If depth is at max (far plane), check neighbors
        if (centerDepth > 0.999) { // Use a threshold close to 1.0
          float minNeighborDepth = 1.0;
 
          // Sample neighbors
          minNeighborDepth = min(minNeighborDepth, readDepth(tDepth, vUv + vec2(texelSize.x, 0.0)));
          minNeighborDepth = min(minNeighborDepth, readDepth(tDepth, vUv - vec2(texelSize.x, 0.0)));
          minNeighborDepth = min(minNeighborDepth, readDepth(tDepth, vUv + vec2(0.0, texelSize.y)));
          minNeighborDepth = min(minNeighborDepth, readDepth(tDepth, vUv - vec2(0.0, texelSize.y)));
 
          // If any neighbor is not at the far plane, use the minimum neighbor depth
          if (minNeighborDepth < 0.999) {
            finalDepth = minNeighborDepth;
          }
        }

        // Adjust contrast to make middle-range depths more visible
        float displayDepth = pow(finalDepth, 0.9);  // Values less than 1 will boost mid-range visibility
 
        // Output grayscale (inverted)
        gl_FragColor = vec4(vec3(1.0 - displayDepth), 1.0);
      }
    `,
    uniforms: {
      tDepth: {
        value: depthTarget.depthTexture,
      },
      cameraNear: {
        value: cameraNear,
      },
      cameraFar: {
        value: cameraFar,
      },
      minDepth: {
        value: 100.0,
      },
      maxDepth: {
        value: 500.0,
      },
      u_useLogarithmicDepth: {
        value: useLogarithmicDepth,
      },
      texelSize: {
        value: new Vector2(1.0 / RENDERER_WIDTH, 1.0 / RENDERER_HEIGHT),
      },
    },
  });

  const planeGeometry = new PlaneGeometry(2, 2);
  const depthMesh = new Mesh(planeGeometry, shaderMaterial);
  const depthScene = new Scene();
  depthScene.add(depthMesh);

  return {
    depthTarget,
    depthScene,
    depthCamera,
    shaderMaterial,
  };
}

function getDepthMatrix(
  depthBuffer: TypedArray,
  width: number,
  height: number,
  camera: PerspectiveCamera
): number[][] {
  const matrix: number[][] = [];

  // Build matrix from bottom to top (flip vertically)
  for (let y = height - 1; y >= 0; y--) {
    const row: number[] = [];
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const r = depthBuffer[idx * 4];
      const g = depthBuffer[idx * 4 + 1];
      const b = depthBuffer[idx * 4 + 2];
      const a = depthBuffer[idx * 4 + 3];

      const relativeLuminance = 0.299 * r + 0.587 * g + 0.114 * b;
      const normalizedDepth = relativeLuminance / 255.0;

      // Convert NDC depth to actual distance from camera
      // The depth buffer contains values in [0, 1] range where:
      // 0 = near plane, 1 = far plane
      // We need to convert this to actual distance in world units

      // For perspective projection, the conversion is:
      // distance = (camera.near * camera.far) / (camera.far - (camera.far - camera.near) * ndcDepth)
      // TODO: how can we access metric units instead of relative?
      // const distance =
      //   (camera.near * camera.far) / (camera.far - (camera.far - camera.near) * normalizedDepth);

      row.push(normalizedDepth);
    }
    matrix.push(row);
  }

  return matrix;
}

function getDepthMatrixStats(depthMatrix: number[][]) {
  let minDistance = Infinity;
  let maxDistance = -Infinity;
  let totalDistance = 0;
  const totalPixels = depthMatrix.length * depthMatrix[0].length;

  for (const row of depthMatrix) {
    for (const distance of row) {
      minDistance = Math.min(minDistance, distance);
      maxDistance = Math.max(maxDistance, distance);
      totalDistance += distance;
    }
  }

  return {
    minDistance: minDistance === Infinity ? 0 : minDistance,
    maxDistance: maxDistance === -Infinity ? 0 : maxDistance,
    avgDistance: totalDistance / totalPixels,
  };
}

async function loadPath(currentPathNumber: number, csvUrl: string) {
  try {
    // First check if the file exists without fetching its contents
    const checkResponse = await fetch(csvUrl, { method: 'HEAD' });

    // Invalid responses should be ignored
    // and skipped to the next path.
    if (!checkResponse.ok) {
      console.log(`Invalid response was received for Path #${currentPathNumber} (${csvUrl})`);
      updatePathStatus(currentPathNumber, PATH_STATUS.DISCARDED, csvUrl, 'no valid data');
      return [];
    }

    // If file exists, proceed with loading
    const response = await fetch(csvUrl);
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
        csvUrl,
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
      updatePathStatus(currentPathNumber, PATH_STATUS.DISCARDED, csvUrl, 'no points');
      return [];
    }

    return newPathData;
  } catch (e: any) {
    console.error(`Error while loading Path #${currentPathNumber}:`, e);
    updatePathStatus(currentPathNumber, PATH_STATUS.DISCARDED, csvUrl, `error: ${e.message}`);
    return [];
  }
}

async function animate(
  tiles: TilesRenderer,
  raycaster: Raycaster,
  renderer: WebGLRenderer,
  camera: PerspectiveCamera,
  scene: Scene,
  depthTarget: WebGLRenderTarget,
  depthCamera: OrthographicCamera,
  depthScene: Scene,
  totalPathsCount: number,
  currentPathIndex: number,
  currentPathNumber: number,
  csvUrl: string,
  currentPathData: Point[],
  currentFrameIndex: number,
  currentFrameRetryCount: number
) {
  // Once the path has completed,
  // try to load the next path,
  // exiting if there are no more paths to be processed.
  if (currentFrameIndex >= currentPathData.length) {
    console.log(`Path ${currentPathNumber} complete.`);
    updatePathStatus(currentPathNumber, PATH_STATUS.COMPLETED, csvUrl);

    if (currentPathIndex >= totalPathsCount - 1) {
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

  // If tiles aren't loaded yet,
  // retry the frame,
  // unless we've exceeded the per-frame retry limit.
  if (tiles.loadProgress < 1 || !tilesLoading) {
    if (currentFrameRetryCount >= MAX_NUMBER_OF_FRAME_RETRIES) {
      console.warn(
        `Tiles not loaded after ${MAX_NUMBER_OF_FRAME_RETRIES} retries, skipping frame #${currentFrameIndex} for Path #${currentPathNumber}`
      );
      requestAnimationFrame(() =>
        animate(
          tiles,
          raycaster,
          renderer,
          camera,
          scene,
          depthTarget,
          depthCamera,
          depthScene,
          totalPathsCount,
          currentPathIndex,
          currentPathNumber,
          csvUrl,
          currentPathData,
          currentFrameIndex + 1,
          0
        )
      );
      return;
    }

    console.warn(
      `Tiles not loaded yet, retrying frame #${
        currentFrameIndex + 1
      } for Path #${currentPathNumber}...`
    );
    requestAnimationFrame(() =>
      animate(
        tiles,
        raycaster,
        renderer,
        camera,
        scene,
        depthTarget,
        depthCamera,
        depthScene,
        totalPathsCount,
        currentPathIndex,
        currentPathNumber,
        csvUrl,
        currentPathData,
        currentFrameIndex,
        currentFrameRetryCount + 1
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
    // TODO: ignore discard to maximize number of frames per path?
    if (
      consecutiveMissingIntersections >= MAX_CONSECUTIVE_MISSING_INTERSECTIONS &&
      SKIP_PATH_AFTER_EXCESSIVE_MISSING_INTERSECTIONS
    ) {
      console.log(
        `${MAX_CONSECUTIVE_MISSING_INTERSECTIONS} consecutive missing intersections for path ${currentPathNumber}, skipping to next path`
      );
      updatePathStatus(
        currentPathNumber,
        PATH_STATUS.DISCARDED,
        csvUrl,
        `${MAX_CONSECUTIVE_MISSING_INTERSECTIONS} consecutive missing intersections`
      );

      if (currentPathIndex >= totalPathsCount - 1) {
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
        raycaster,
        renderer,
        camera,
        scene,
        depthTarget,
        depthCamera,
        depthScene,
        totalPathsCount,
        currentPathIndex,
        currentPathNumber,
        csvUrl,
        currentPathData,
        currentFrameIndex + 1,
        0
      )
    );
    return;
  }

  // Discard the entire path if the intersection is too close to the ground
  // TODO: ignore discard to maximize number of frames per path?
  if (lookAtPoint.distance < MIN_ABOVE_GROUND_DISTANCE && SKIP_PATH_AFTER_COLLISION_WITH_GROUND) {
    console.log(
      `Intersection too close (${lookAtPoint.distance.toFixed(
        2
      )}m) for path ${currentPathNumber}, skipping to next path`
    );
    updatePathStatus(
      currentPathNumber,
      PATH_STATUS.DISCARDED,
      csvUrl,
      `intersection too close (${lookAtPoint.distance.toFixed(2)}m)`
    );

    if (currentPathIndex >= totalPathsCount - 1) {
      console.log('All paths complete');
      console.log(MESSAGE_TYPES.PROCESSING_COMPLETE);
    }

    currentPathPending = false;

    return;
  }

  // Discard the entire path if the intersection is too far from the ground
  // TODO: ignore discard to maximize number of frames per path?
  if (lookAtPoint.distance > MAX_ABOVE_GROUND_DISTANCE && SKIP_PATH_AFTER_UNREALISTIC_HEIGHT) {
    console.log(
      `Intersection too far (${lookAtPoint.distance.toFixed(
        2
      )}m) for path ${currentPathNumber}, skipping to next path`
    );
    updatePathStatus(
      currentPathNumber,
      PATH_STATUS.DISCARDED,
      csvUrl,
      `intersection too far (${lookAtPoint.distance.toFixed(2)}m)`
    );

    if (currentPathIndex >= totalPathsCount - 1) {
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

  // First render the normal scene and take the screenshot
  renderer.setRenderTarget(null);
  renderer.render(scene, camera);
  console.log(`${MESSAGE_TYPES.REGULAR_FRAME_READY}_${JSON.stringify(metadata)}`);

  // Wait for Puppeteer to capture the frame,
  // unless we hit a timeout,
  // in which case we continue to the next frame.
  try {
    await new Promise((resolve) => {
      const messageHandler = (event: MessageEvent) => {
        // TODO: change to event.type === 'console' && event.detail?.text === MESSAGE_TYPES.REGULAR_FRAME_CAPTURED
        if (event.data === MESSAGE_TYPES.REGULAR_FRAME_CAPTURED) {
          window.removeEventListener('message', messageHandler);
          resolve(undefined);
        }
      };

      // Send message to Puppeteer
      window.addEventListener('message', messageHandler);
      window.postMessage(MESSAGE_TYPES.REGULAR_FRAME_READY, '*');

      // Make sure to resolve the Promise,
      // even if the message is never received.
      setTimeout(() => {
        // TODO: change to 'console' instead of 'message'
        window.removeEventListener('message', messageHandler);
        console.warn('Regular frame capture confirmation timed out');
        resolve(undefined);
      }, MESSAGE_DELAY_IN_MS);
    });
  } catch (error) {
    console.error('Error during regular frame capture:', error);

    requestAnimationFrame(() =>
      animate(
        tiles,
        raycaster,
        renderer,
        camera,
        scene,
        depthTarget,
        depthCamera,
        depthScene,
        totalPathsCount,
        currentPathIndex,
        currentPathNumber,
        csvUrl,
        currentPathData,
        currentFrameIndex + 1,
        0
      )
    );

    return;
  }

  // Then render depth view and take screenshot
  // 1. Render scene to depth texture
  renderer.setRenderTarget(depthTarget);
  renderer.render(scene, camera);

  // Read depth buffer directly from the depth texture before rendering visualization
  const width = depthTarget.width;
  const height = depthTarget.height;
  const depthBuffer = new Uint8Array(width * height * 4);
  renderer.readRenderTargetPixels(depthTarget, 0, 0, width, height, depthBuffer);
  const depthMatrix = getDepthMatrix(depthBuffer, width, height, camera);
  const depthStats = getDepthMatrixStats(depthMatrix);

  // 2. Render depth visualization quad to canvas
  renderer.setRenderTarget(null);

  // TODO: these values are fixed when shaderMaterial is defined, keeping for bookeepping's sake
  // shaderMaterial.uniforms.cameraNear.value = camera.near;
  // shaderMaterial.uniforms.cameraFar.value = camera.far;
  // shaderMaterial.uniforms.texelSize.value.set(1.0 / RENDERER_WIDTH, 1.0 / RENDERER_HEIGHT);
  // shaderMaterial.uniforms.minDepth.value = minDepth;
  // shaderMaterial.uniforms.maxDepth.value = maxDepth;

  renderer.render(depthScene, depthCamera);

  // Log the depth matrix data with statistics
  console.log(
    `${MESSAGE_TYPES.DEPTH_MATRIX_READY}_${JSON.stringify({
      pathNumber: currentPathNumber,
      frameIndex: currentFrameIndex,
      depthMatrix,
      depthStats,
    })}`
  );

  console.log(`${MESSAGE_TYPES.DEPTH_FRAME_READY}_${JSON.stringify(metadata)}`);

  // Wait for Puppeteer to capture the frame,
  // unless we hit a timeout,
  // in which case we continue to the next frame.
  try {
    await new Promise((resolve) => {
      const messageHandler = (event: MessageEvent) => {
        // TODO: change to event.type === 'console' && event.detail?.text === MESSAGE_TYPES.REGULAR_FRAME_CAPTURED
        if (event.data === MESSAGE_TYPES.DEPTH_FRAME_CAPTURED) {
          window.removeEventListener('message', messageHandler);
          resolve(undefined);
        }
      };

      // Send message to Puppeteer
      window.addEventListener('message', messageHandler);
      window.postMessage(MESSAGE_TYPES.DEPTH_FRAME_READY, '*');

      // Make sure to resolve the Promise,
      // even if the message is never received.
      setTimeout(() => {
        // TODO: change to 'console' instead of 'message'
        window.removeEventListener('message', messageHandler);
        console.warn('Depth frame capture confirmation timed out');
        resolve(undefined);
      }, MESSAGE_DELAY_IN_MS);
    });
  } catch (error) {
    console.error('Error during depth frame capture:', error);
  } finally {
    requestAnimationFrame(() =>
      animate(
        tiles,
        raycaster,
        renderer,
        camera,
        scene,
        depthTarget,
        depthCamera,
        depthScene,
        totalPathsCount,
        currentPathIndex,
        currentPathNumber,
        csvUrl,
        currentPathData,
        currentFrameIndex + 1,
        0
      )
    );
  }
}

async function main(
  cameraFov: number = 35,
  cameraNear: number = 1,
  cameraFar: number = 16000000,
  depthFar: number = 700,
  useLogarithmicDepth: boolean = true
) {
  const scene = new Scene();
  const raycaster = new Raycaster();

  const renderer = new WebGLRenderer({ antialias: true });
  renderer.setClearColor(0x87ceeb);
  document.body.appendChild(renderer.domElement);

  // TODO: is the fov fixed? is the near plane fixed? is the far plane fixed?
  const camera = new PerspectiveCamera(
    cameraFov,
    window.innerWidth / window.innerHeight,
    cameraNear,
    cameraFar
  );
  const tiles = new TilesRenderer();

  const { depthTarget, depthScene, depthCamera } = setupDepthRendering(
    cameraNear,
    depthFar,
    useLogarithmicDepth
  );

  // Fetch all query parameters from URL
  const urlParams = new URLSearchParams(window.location.search);

  // Check and validate the Cesium token
  const token = urlParams.get('token') || null;

  if (!token) {
    console.error('No token provided');
    return;
  }

  // TODO: how do we know that the provided token is valid at runtime?
  reinstantiateTiles(tiles, camera, renderer, scene, token);

  // Check and validate the csvUrls
  const rawCsvUrls = urlParams.get('csvUrls') || null;

  if (!rawCsvUrls) {
    console.error('No CSV URLs provided');
    return;
  }

  const csvUrls = rawCsvUrls.split(',');

  if (!isArrayOfString(csvUrls) || csvUrls.length === 0) {
    console.error('No valid CSV URLs provided');
    return;
  }

  onWindowResize(camera, renderer, depthTarget);
  window.addEventListener('resize', () => onWindowResize(camera, renderer, depthTarget), false);
  tiles.addEventListener('tiles-load-start', () => {
    console.log('Tiles loaded');
    tilesLoading = true;
  });

  let atLeastOnePathProcessed = false;

  for (let currentPathIndex = 0; currentPathIndex < csvUrls.length; currentPathIndex++) {
    // Reset all state for new path
    tilesLoading = false;
    consecutiveMissingIntersections = 0;
    currentPathPending = true;

    const currentCsvUrl = csvUrls[currentPathIndex];
    const currentPathNumber = getPathNumberFromCsvUrl(currentCsvUrl);
    const currentPathData = await loadPath(currentPathNumber, currentCsvUrl);

    if (currentPathData.length > 0) {
      atLeastOnePathProcessed = true;

      animate(
        tiles,
        raycaster,
        renderer,
        camera,
        scene,
        depthTarget,
        depthCamera,
        depthScene,
        csvUrls.length,
        currentPathIndex,
        currentPathNumber,
        currentCsvUrl,
        currentPathData,
        0,
        0
      );

      // Wait for the current path to complete before moving to the next one
      while (currentPathPending) {
        await new Promise((_resolve) => setTimeout(_resolve, PATH_WAIT_DELAY_IN_MS));
      }
    } else {
      currentPathPending = false;
    }
  }

  // Make sure we send the processing complete message
  if (!atLeastOnePathProcessed) {
    console.log('All paths complete');
    console.log(MESSAGE_TYPES.PROCESSING_COMPLETE);
  }
}

main();
