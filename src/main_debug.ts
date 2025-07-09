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
  TextureLoader,
  RepeatWrapping,
  LinearFilter,
  LinearMipMapLinearFilter,
  RedFormat,
  NoColorSpace,
  HalfFloatType,
  NoToneMapping,
} from 'three';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader';
import {
  EffectComposer,
  EffectPass,
  NormalPass,
  RenderPass,
  ToneMappingEffect,
  ToneMappingMode,
} from 'postprocessing';
import {
  AerialPerspectiveEffect,
  getSunDirectionECEF,
  PrecomputedTexturesLoader,
} from '@takram/three-atmosphere';
import { CloudsEffect } from '@takram/three-clouds';
import {
  CLOUD_SHAPE_DETAIL_TEXTURE_SIZE,
  CLOUD_SHAPE_TEXTURE_SIZE,
} from '@takram/three-clouds';
import {
  createData3DTextureLoaderClass,
  parseUint8Array,
  STBNLoader,
} from '@takram/three-geospatial';
import { DitheringEffect, LensFlareEffect } from '@takram/three-geospatial-effects';
import { OrbitControls } from 'three-stdlib';

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

const RENDERER_WIDTH = window.innerWidth;
const RENDERER_HEIGHT = window.innerHeight;

const DEFAULT_CAMERA_FAR_IN_METERS = 100000;
const DEFAULT_CAMERA_NEAR_IN_METERS = 1;
const DEFAULT_CAMERA_FOV_IN_DEGREES = 35;

let composer: EffectComposer;
let controls: OrbitControls;

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

function decodeDepthFromGrayscaleRGBA(r: number, g: number, b: number, a: number) {
  return (r + g / 256 + b / 65536 + a / 16777216) / 255;
}

function isArrayOfString(value: any) {
  if (!Array.isArray(value)) {
    return false;
  }
  return !value.some((item) => typeof item !== 'string');
}

function onWindowResize(camera: PerspectiveCamera, renderer: WebGLRenderer, tiles: TilesRenderer) {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  composer.setSize(window.innerWidth, window.innerHeight);

  // Keep 3d-tiles-renderer in sync so error calculation uses correct resolution
  tiles.setResolutionFromRenderer(camera, renderer);
  tiles.setCamera(camera);
}

function getLookAtPoint(tiles: TilesRenderer, camera: PerspectiveCamera, raycaster: Raycaster) {
  raycaster.setFromCamera({ x: 0, y: 0 } as Vector2, camera);
  const intersects = raycaster.intersectObject(tiles.group, true);

  // TODO: what if there are multiple intersections?
  if (intersects.length > 0) {
    const point = intersects[0].point;
    // Convert world position to local position in tiles coordinate system
    const mat = (tiles.group as any).matrixWorld.clone().invert();
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
    const mat = (tiles.group as any).matrixWorld.clone().invert();
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

  // Inform tiles renderer about viewport resolution once so error metric is valid
  tiles.setResolutionFromRenderer(camera, renderer);
  tiles.setCamera(camera);
}

function setupDepthRendering(cameraNear: number, cameraFar: number) {
  // Create render target with depth texture
  const depthTarget = new WebGLRenderTarget(RENDERER_WIDTH, RENDERER_HEIGHT);
  depthTarget.texture.minFilter = NearestFilter;
  depthTarget.texture.magFilter = NearestFilter;
  depthTarget.depthTexture = new DepthTexture(RENDERER_WIDTH, RENDERER_HEIGHT);
  depthTarget.depthTexture.type = FloatType;

  // Render target that will receive the packed RGBA depth
  const packedDepthTarget = new WebGLRenderTarget(RENDERER_WIDTH, RENDERER_HEIGHT);
  packedDepthTarget.texture.minFilter = NearestFilter;
  packedDepthTarget.texture.magFilter = NearestFilter;

  // Setup post-processing for depth packing (metric depth → RGBA8)
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
      uniform vec2 texelSize;

      // Returns depth normalised to 0-1 range (0 = near, 1 = cameraFar)
      float readDepth( sampler2D depthSampler, vec2 coord ) {
          float fragCoordZ = texture2D( depthSampler, coord ).x;
          float viewZ      = perspectiveDepthToViewZ( fragCoordZ, cameraNear, cameraFar );
          float linear     = clamp( -viewZ, 0.0, cameraFar );   // metres
          return linear / cameraFar;                            // 0-1
      }

      void main() {
        float centerDepth = readDepth( tDepth, vUv );
        float finalDepth  = centerDepth;

        // Gap-filling: if depth equals far-plane, try use closest neighbour that has data
        if ( centerDepth > 0.999 ) {
          float minNeighbour = 1.0;
          minNeighbour = min( minNeighbour, readDepth( tDepth, vUv + vec2( texelSize.x, 0.0 ) ) );
          minNeighbour = min( minNeighbour, readDepth( tDepth, vUv - vec2( texelSize.x, 0.0 ) ) );
          minNeighbour = min( minNeighbour, readDepth( tDepth, vUv + vec2( 0.0, texelSize.y ) ) );
          minNeighbour = min( minNeighbour, readDepth( tDepth, vUv - vec2( 0.0, texelSize.y ) ) );
          if ( minNeighbour < 0.999 ) {
            finalDepth = minNeighbour;
          }
        }

        // Pack the 0-1 depth into RGBA8 (loss-less when saved as PNG)
        gl_FragColor = packDepthToRGBA( finalDepth );
      }
    `,
    uniforms: {
      tDepth: { value: depthTarget.depthTexture },
      cameraNear: { value: cameraNear },
      cameraFar: { value: cameraFar },
      texelSize: { value: new Vector2(1.0 / RENDERER_WIDTH, 1.0 / RENDERER_HEIGHT) },
    },
  });

  const planeGeometry = new PlaneGeometry(2, 2);
  const depthMesh = new Mesh(planeGeometry, shaderMaterial);
  const depthScene = new Scene();
  depthScene.add(depthMesh);

  return {
    depthTarget,
    packedDepthTarget,
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
  packedDepthTarget: WebGLRenderTarget,
  depthCamera: OrthographicCamera,
  depthScene: Scene,
  totalPathsCount: number,
  currentPathIndex: number,
  currentPathNumber: number,
  csvUrl: string,
  currentPathData: Point[],
  currentFrameIndex: number,
  currentFrameRetryCount: number,
  debugDepthFrame: boolean,
  roundDepthMatrix: number
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

    return;
  }

  // Re-position camera for current point
  const currentPoint = currentPathData[currentFrameIndex];
  // @ts-ignore
  (tiles as any).setLatLonToYUp(currentPoint.lat * MathUtils.DEG2RAD, currentPoint.lng * MathUtils.DEG2RAD);
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
  if (tiles.loadProgress < 1) {
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
          packedDepthTarget,
          depthCamera,
          depthScene,
          totalPathsCount,
          currentPathIndex,
          currentPathNumber,
          csvUrl,
          currentPathData,
          currentFrameIndex + 1,
          0,
          debugDepthFrame,
          roundDepthMatrix
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
        packedDepthTarget,
        depthCamera,
        depthScene,
        totalPathsCount,
        currentPathIndex,
        currentPathNumber,
        csvUrl,
        currentPathData,
        currentFrameIndex,
        currentFrameRetryCount + 1,
        debugDepthFrame,
        roundDepthMatrix
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
    requestAnimationFrame(() =>
      animate(
        tiles,
        raycaster,
        renderer,
        camera,
        scene,
        depthTarget,
        packedDepthTarget,
        depthCamera,
        depthScene,
        totalPathsCount,
        currentPathIndex,
        currentPathNumber,
        csvUrl,
        currentPathData,
        currentFrameIndex + 1,
        0,
        debugDepthFrame,
        roundDepthMatrix
      )
    );
    return;
  }

  // Discard the entire path if the intersection is too close to the ground
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

    return;
  }

  // Discard the entire path if the intersection is too far from the ground
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

    return;
  }

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
  composer.render();
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
        packedDepthTarget,
        depthCamera,
        depthScene,
        totalPathsCount,
        currentPathIndex,
        currentPathNumber,
        csvUrl,
        currentPathData,
        currentFrameIndex + 1,
        0,
        debugDepthFrame,
        roundDepthMatrix
      )
    );

    return;
  }

  // -------------------------------------------------------------
  // Depth rendering is disabled for debugging (black output fix)
  // -------------------------------------------------------------
  requestAnimationFrame(() =>
    animate(
      tiles,
      raycaster,
      renderer,
      camera,
      scene,
      depthTarget,
      packedDepthTarget,
      depthCamera,
      depthScene,
      totalPathsCount,
      currentPathIndex,
      currentPathNumber,
      csvUrl,
      currentPathData,
      currentFrameIndex + 1,
      0,
      debugDepthFrame,
      roundDepthMatrix
    )
  );

  return;
}

function parseQueryParams() {
  // Fetch all query parameters from URL
  const urlParams = new URLSearchParams(window.location.search);

  // Check and validate the Cesium token
  const token = urlParams.get('token') || null;

  if (!token) {
    throw new Error('No token provided');
  }

  // Check and validate the csvUrls
  const rawCsvUrls = urlParams.get('csvUrls') || null;

  if (!rawCsvUrls) {
    throw new Error('No CSV URLs provided');
  }

  const csvUrls = rawCsvUrls.split(',');

  if (!isArrayOfString(csvUrls) || csvUrls.length === 0) {
    throw new Error('No valid CSV URLs provided');
  }

  const cameraFov = Number(urlParams.get('cameraFov')) || DEFAULT_CAMERA_FOV_IN_DEGREES;

  if (isNaN(cameraFov) || cameraFov <= 0) {
    throw new Error('Invalid camera fov value');
  }

  const cameraNear = Number(urlParams.get('cameraNear')) || DEFAULT_CAMERA_NEAR_IN_METERS;

  if (isNaN(cameraNear) || cameraNear <= 0) {
    throw new Error('Invalid camera near value');
  }

  const cameraFar = Number(urlParams.get('cameraFar')) || DEFAULT_CAMERA_FAR_IN_METERS;

  if (isNaN(cameraFar) || cameraFar <= 0) {
    throw new Error('Invalid camera far value');
  }

  if (cameraNear >= cameraFar) {
    throw new Error('Camera near value must be less than camera far value');
  }

  const debugDepthFrame = Boolean(urlParams.get('debugDepthFrame')) || false;

  const roundDepthMatrix = Number(urlParams.get('roundDepthMatrix')) || 0;

  if (isNaN(roundDepthMatrix) || roundDepthMatrix < 0) {
    throw new Error('Invalid round depth matrix value');
  }

  return { token, csvUrls, cameraFar, cameraNear, cameraFov, debugDepthFrame, roundDepthMatrix };
}

function main() {
  // Fetch all query parameters from URL
  const { token, csvUrls, cameraFar, cameraNear, cameraFov, debugDepthFrame, roundDepthMatrix } =
    parseQueryParams();

  const scene = new Scene();
  const raycaster = new Raycaster();

  // Configure renderer similar to the reference implementation so that HDR
  // values produced by the atmosphere & clouds effects are not clamped which
  // resulted in an overall very dark output.
  const renderer = new WebGLRenderer({
    antialias: true,
    logarithmicDepthBuffer: false,
  });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(0x87ceeb);

  // Keep the renderer output in HDR and apply tone-mapping later in the
  // post-processing chain. Increasing the exposure drastically brightens the
  // scene which otherwise appears far too dark.
  renderer.toneMapping = NoToneMapping;
  renderer.toneMappingExposure = 10;

  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  const camera = new PerspectiveCamera(cameraFov, window.innerWidth / window.innerHeight, cameraNear, cameraFar);
  const tiles = new TilesRenderer();

  // Orbit controls for debugging
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.minDistance = 5;
  controls.maxDistance = 5000;

  // Use a floating-point framebuffer so the AGX tone-mapping pass can work with
  // the full radiance range coming from the atmosphere & clouds effects.
  composer = new EffectComposer(renderer, {
    frameBufferType: HalfFloatType,
    multisampling: 0,
  });
  composer.addPass(new RenderPass(scene, camera));

  // ----------------------------------
  // Atmosphere & clouds postprocessing
  // ----------------------------------
  // Setup sun and effects AFTER camera is initialised
  const sunDirection = new Vector3();
  getSunDirectionECEF(new Date(), sunDirection);

  const aerialPerspective = new AerialPerspectiveEffect(camera);
  aerialPerspective.sky = true;
  aerialPerspective.sunIrradiance = true;
  aerialPerspective.skyIrradiance = true;
  aerialPerspective.sunDirection.copy(sunDirection);

  const clouds = new CloudsEffect(camera);
  clouds.coverage = 0.4; // match reference so clouds are actually visible
  clouds.localWeatherVelocity.set(0.001, 0);
  clouds.sunDirection.copy(sunDirection);

  const normalPass = new NormalPass(scene, camera);
  aerialPerspective.normalBuffer = normalPass.texture;

  composer.addPass(normalPass);
  composer.addPass(new EffectPass(camera, clouds, aerialPerspective));
  composer.addPass(
    new EffectPass(
      camera,
      new LensFlareEffect(),
      new ToneMappingEffect({ mode: ToneMappingMode.AGX }),
      new DitheringEffect()
    )
  );

  // --------------------------------------------------
  // Load pre-computed lookup textures (sky & clouds)
  // --------------------------------------------------

  const basePath = import.meta.env.BASE_URL || '/';

  const assetsReady = new Promise<void>((resolve) => {
    let remaining = 6; // 1 atmosphere + 5 cloud related

    const done = () => {
      remaining -= 1;
      if (remaining === 0) {
        resolve();
      }
    };

    // Atmosphere LUTs
    new PrecomputedTexturesLoader().load(basePath + 'assets/atmosphere', (textures: any) => {
      Object.assign(aerialPerspective, textures);
      Object.assign(clouds, textures);
      done();
    });

    // Clouds local weather map
    new TextureLoader().load(basePath + 'assets/clouds/local_weather.png', (texture) => {
      texture.minFilter = LinearMipMapLinearFilter;
      texture.magFilter = LinearFilter;
      texture.wrapS = RepeatWrapping;
      texture.wrapT = RepeatWrapping;
      texture.colorSpace = NoColorSpace as any;
      texture.needsUpdate = true;
      clouds.localWeatherTexture = texture;
      done();
    });

    // Clouds shape volume
    new (createData3DTextureLoaderClass(parseUint8Array, {
      width: CLOUD_SHAPE_TEXTURE_SIZE,
      height: CLOUD_SHAPE_TEXTURE_SIZE,
      depth: CLOUD_SHAPE_TEXTURE_SIZE,
    }))().load(basePath + 'assets/clouds/shape.bin', (texture: any) => {
      texture.format = RedFormat;
      texture.minFilter = LinearFilter;
      texture.magFilter = LinearFilter;
      texture.wrapS = texture.wrapT = texture.wrapR = RepeatWrapping;
      texture.colorSpace = NoColorSpace as any;
      texture.needsUpdate = true;
      clouds.shapeTexture = texture;
      done();
    });

    // Clouds shape detail volume
    new (createData3DTextureLoaderClass(parseUint8Array, {
      width: CLOUD_SHAPE_DETAIL_TEXTURE_SIZE,
      height: CLOUD_SHAPE_DETAIL_TEXTURE_SIZE,
      depth: CLOUD_SHAPE_DETAIL_TEXTURE_SIZE,
    }))().load(basePath + 'assets/clouds/shape_detail.bin', (texture: any) => {
      texture.format = RedFormat;
      texture.minFilter = LinearFilter;
      texture.magFilter = LinearFilter;
      texture.wrapS = texture.wrapT = texture.wrapR = RepeatWrapping;
      texture.colorSpace = NoColorSpace as any;
      texture.needsUpdate = true;
      clouds.shapeDetailTexture = texture;
      done();
    });

    // Clouds turbulence map
    new TextureLoader().load(basePath + 'assets/clouds/turbulence.png', (texture) => {
      texture.minFilter = LinearMipMapLinearFilter;
      texture.magFilter = LinearFilter;
      texture.wrapS = RepeatWrapping;
      texture.wrapT = RepeatWrapping;
      texture.colorSpace = NoColorSpace as any;
      texture.needsUpdate = true;
      clouds.turbulenceTexture = texture;
      done();
    });

    // STBN blue-noise (shared by both effects)
    new STBNLoader().load(basePath + 'assets/core/stbn.bin', (texture: any) => {
      aerialPerspective.stbnTexture = texture;
      clouds.stbnTexture = texture;
      done();
    });
  });

  // --------------------------------------------------
  // Assets are ready – continue normal execution
  // --------------------------------------------------

  assetsReady.then(() => {
    const { depthTarget, packedDepthTarget, depthScene, depthCamera } = setupDepthRendering(
      cameraNear,
      cameraFar
    );

    // Place camera over a default location (e.g. New York) and orient tiles
    const initialLat = 40.7128; // degrees
    const initialLon = -74.0060;
    const initialAlt = 1200; // metres above ground

    (tiles as any).setLatLonToYUp(initialLat * MathUtils.DEG2RAD, initialLon * MathUtils.DEG2RAD);
    camera.position.set(0, initialAlt, 0);
    camera.lookAt(new Vector3(0, 0, 0));
    camera.updateMatrixWorld();

    // TODO: how do we know that the provided token is valid at runtime?
    reinstantiateTiles(tiles, camera, renderer, scene, token);

    window.addEventListener('resize', () => onWindowResize(camera, renderer, tiles));

    // simple render loop (start only after all textures are in place)
    const renderLoop = () => {
      controls.update();
      tiles.update();
      composer.render();
    };

    renderer.setAnimationLoop(renderLoop);
  });
}

main();
