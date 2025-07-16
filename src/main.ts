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
  DataTexture,
  LinearFilter,
  RGBAFormat,
  UnsignedByteType,
  Matrix4,
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

const DEFAULT_RENDERER_WIDTH = 512;
const DEFAULT_RENDERER_HEIGHT = 384;
const DEFAULT_CAMERA_FAR_IN_METERS = 100000;
const DEFAULT_CAMERA_NEAR_IN_METERS = 1;
const DEFAULT_CAMERA_FOV_IN_DEGREES = 35;

// Tolerance when checking occlusion in camera B
const OVERLAP_EPSILON_METERS = 0.5;

// Global flags
let tilesLoading = false;
let consecutiveMissingIntersections = 0;
let currentPathPending = false;

// Collection of per-frame data used to compute view-overlap for the current path.
// It is reset at the start of every path.
let currentPathFrames: any[] = [];

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

// Type for the data we need to compute the overlap matrix
type FrameInfo = {
  dataTexture: DataTexture; // packed RGBA8 depth
  camera: PerspectiveCamera; // cloned camera for this frame (contains matrices)
  width: number;
  height: number;
  cameraFar: number;
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

function onWindowResize(
  width: number,
  height: number,
  camera: PerspectiveCamera,
  renderer: WebGLRenderer,
  depthTarget: WebGLRenderTarget,
  packedDepthTarget: WebGLRenderTarget
) {
  camera.aspect = 1;
  renderer.setSize(width, height);
  depthTarget.setSize(width, height);
  packedDepthTarget.setSize(width, height);

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

function setupDepthRendering(width: number, height: number, cameraNear: number, cameraFar: number) {
  // Create render target with depth texture
  const depthTarget = new WebGLRenderTarget(width, height);
  depthTarget.texture.minFilter = NearestFilter;
  depthTarget.texture.magFilter = NearestFilter;
  depthTarget.depthTexture = new DepthTexture(width, height);
  depthTarget.depthTexture.type = FloatType;

  // Render target that will receive the packed RGBA depth
  const packedDepthTarget = new WebGLRenderTarget(width, height);
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
      texelSize: { value: new Vector2(1.0 / width, 1.0 / height) },
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

function setupOverlapResources(width: number, height: number) {
  const overlapScene = new Scene();
  const overlapCamera = new OrthographicCamera(-1, 1, 1, -1, 0, 1);

  const overlapRenderTarget = new WebGLRenderTarget(width, height);
  overlapRenderTarget.texture.minFilter = LinearFilter;
  overlapRenderTarget.texture.magFilter = LinearFilter;

  const material = new ShaderMaterial({
    uniforms: {
      tDepthA: { value: null },
      tDepthB: { value: null },
      invPA: { value: new Matrix4() },
      invVA: { value: new Matrix4() },
      PB: { value: new Matrix4() },
      VB: { value: new Matrix4() },
      farA: { value: 1 },
      farB: { value: 1 },
      epsilon: { value: OVERLAP_EPSILON_METERS },
    },
    vertexShader: `
      varying vec2 vUv;
      void main(){
        vUv = uv;
        gl_Position = vec4(position.xy, 0.0, 1.0);
      }
    `,
    fragmentShader: `
      precision highp float;
      varying vec2 vUv;
      uniform sampler2D tDepthA;
      uniform sampler2D tDepthB;
      uniform mat4 invPA;
      uniform mat4 invVA;
      uniform mat4 PB;
      uniform mat4 VB;
      uniform float farA;
      uniform float farB;
      uniform float epsilon;

      // Unpack the packed RGBA8 depth value (0-1 range storing linear depth / far)
      float unpackDepth( vec4 c ) {
        return c.r + c.g / 256.0 + c.b / 65536.0 + c.a / 16777216.0;
      }

      void main() {
        // ------------------------------------------------------------------
        // 1. Read linear depth for the current pixel in camera A
        // ------------------------------------------------------------------
        float zNormA = unpackDepth( texture2D( tDepthA, vUv ) );
        if ( zNormA > 0.999 ) {
          discard;                        // no geometry at far plane
        }

        // Convert to metric depth (metres)
        float depthA = zNormA * farA;

        // ------------------------------------------------------------------
        // 2. Reconstruct the view-space position for camera A
        // ------------------------------------------------------------------
        // Build a ray direction from the camera through this pixel
        vec2 ndc = vUv * 2.0 - 1.0;         // Normalised device coordinates
        vec4 clipNear = vec4( ndc, -1.0, 1.0 );
        vec4 viewNear = invPA * clipNear;   // to view space
        viewNear /= viewNear.w;
        vec3 rayDir = normalize( viewNear.xyz );

        // View-space position (camera A looks down -Z axis)
        // depthA is positive; rayDir points toward the scene (negative z),
        // so we multiply without flipping the sign.
        vec3 viewPosA = rayDir * depthA;

        // World-space position of this sample
        vec4 worldPos = invVA * vec4( viewPosA, 1.0 );

        // ------------------------------------------------------------------
        // 3. Project the point into camera B
        // ------------------------------------------------------------------
        vec4 viewPosB4 = VB * worldPos;     // view space B
        if ( viewPosB4.z >= 0.0 ) {
          discard;                          // behind camera B
        }

        vec4 clipB = PB * viewPosB4;
        if ( clipB.w <= 0.0 ) {
          discard;
        }

        vec3 ndcB = clipB.xyz / clipB.w;
        if ( abs( ndcB.x ) > 1.0 || abs( ndcB.y ) > 1.0 || ndcB.z < 0.0 || ndcB.z > 1.0 ) {
          discard;                          // outside frustum B
        }

        vec2 uvB = ndcB.xy * 0.5 + 0.5;

        // ------------------------------------------------------------------
        // 4. Depth test in camera B (linear space)
        // ------------------------------------------------------------------
        float zNormB = unpackDepth( texture2D( tDepthB, uvB ) );
        float depthPixelB = zNormB * farB;      // surface depth in B for this pixel
        float depthSampleB = -viewPosB4.z;      // depth of our sample in B

        if ( depthSampleB <= depthPixelB + epsilon ) {
          // Visible from both cameras -> overlap
          gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );
        } else {
          gl_FragColor = vec4( 0.0, 0.0, 0.0, 1.0 );
        }
      }
    `,
  });

  const overlapQuad = new Mesh(new PlaneGeometry(2, 2), material);
  overlapScene.add(overlapQuad);

  return { overlapScene, overlapCamera, overlapQuad, overlapRenderTarget };
}

function gpuOverlap(
  renderer: WebGLRenderer,
  overlapScene: Scene,
  overlapCamera: OrthographicCamera,
  overlapQuad: Mesh,
  overlapRenderTarget: WebGLRenderTarget,
  a: FrameInfo,
  b: FrameInfo
): number {
  if (overlapRenderTarget.width !== a.width || overlapRenderTarget.height !== a.height) {
    throw new Error(
      `Overlap resources not initialized (width: ${overlapRenderTarget.width}, height: ${overlapRenderTarget.height}, expected: ${a.width}x${a.height})`
    );
  }

  const material = overlapQuad.material as ShaderMaterial;
  material.uniforms.tDepthA.value = a.dataTexture;
  material.uniforms.tDepthB.value = b.dataTexture;
  material.uniforms.invPA.value = a.camera.projectionMatrixInverse;
  material.uniforms.invVA.value = a.camera.matrixWorld; // world matrix (== inverse of view)
  material.uniforms.PB.value = b.camera.projectionMatrix;
  material.uniforms.VB.value = b.camera.matrixWorldInverse;
  material.uniforms.farA.value = a.cameraFar;
  material.uniforms.farB.value = b.cameraFar;

  // ------------------------------------------------------------------
  // 1. Try native WebGL2 occlusion query
  // ------------------------------------------------------------------
  const gl: WebGLRenderingContext | WebGL2RenderingContext = renderer.getContext();
  const isWebGL2 = (gl as WebGL2RenderingContext).beginQuery !== undefined;
  let passedSamples: number | null = null;

  if (isWebGL2) {
    const gl2 = gl as WebGL2RenderingContext;
    const query = gl2.createQuery();
    if (query) {
      renderer.setRenderTarget(overlapRenderTarget);
      gl2.beginQuery(gl2.ANY_SAMPLES_PASSED, query);
      renderer.render(overlapScene, overlapCamera);
      gl2.endQuery(gl2.ANY_SAMPLES_PASSED);
      gl2.flush(); // ensure the commands are submitted

      // Busy-wait until the result is ready. In practice this returns quickly.
      while (!gl2.getQueryParameter(query, gl2.QUERY_RESULT_AVAILABLE)) {
        /* spin */
      }
      passedSamples = gl2.getQueryParameter(query, gl2.QUERY_RESULT);
      gl2.deleteQuery(query);
      renderer.setRenderTarget(null);
    }
  }

  // ------------------------------------------------------------------
  // 2. WebGL1 fall-back using EXT_occlusion_query_boolean
  // ------------------------------------------------------------------
  if (passedSamples === null) {
    // Either WebGL2 path was unavailable or failed → try extension
    const ext: any = gl.getExtension('EXT_occlusion_query_boolean');
    if (ext) {
      const queryExt = ext.createQueryEXT();
      renderer.setRenderTarget(overlapRenderTarget);
      ext.beginQueryEXT(ext.ANY_SAMPLES_PASSED_EXT, queryExt);
      renderer.render(overlapScene, overlapCamera);
      ext.endQueryEXT(ext.ANY_SAMPLES_PASSED_EXT);
      (gl as WebGLRenderingContext).flush();

      while (!ext.getQueryObjectEXT(queryExt, ext.QUERY_RESULT_AVAILABLE_EXT)) {
        /* spin */
      }
      passedSamples = ext.getQueryObjectEXT(queryExt, ext.QUERY_RESULT_EXT);
      ext.deleteQueryEXT(queryExt);
      renderer.setRenderTarget(null);
    }
  }
  // ------------------------------------------------------------------
  // 3. Final fall-back → original readPixels implementation
  // ------------------------------------------------------------------
  if (passedSamples === null) {
    renderer.setRenderTarget(overlapRenderTarget);
    renderer.render(overlapScene, overlapCamera);

    const buffer = new Uint8Array(a.width * a.height * 4);
    renderer.readRenderTargetPixels(overlapRenderTarget, 0, 0, a.width, a.height, buffer);
    renderer.setRenderTarget(null);

    let count = 0;
    for (let i = 0; i < buffer.length; i += 4) {
      if (buffer[i] > 128) count++;
    }
    passedSamples = count;
  }

  const totalPixels = a.width * a.height;
  return passedSamples / totalPixels;

  // renderer.setRenderTarget(overlapRenderTarget);
  // renderer.render(overlapScene, overlapCamera);

  // const buffer = new Uint8Array(a.width * a.height * 4);
  // renderer.readRenderTargetPixels(overlapRenderTarget, 0, 0, a.width, a.height, buffer);

  // let overlapCount = 0;
  // for (let i = 0; i < buffer.length; i += 4) {
  //   if (buffer[i] > 128) overlapCount++;
  // }

  // const total = a.width * a.height;
  // renderer.setRenderTarget(null);

  // return overlapCount / total;
}

function computeCovizMatrix(
  frames: FrameInfo[],
  renderer: WebGLRenderer,
  overlapScene: Scene,
  overlapCamera: OrthographicCamera,
  overlapQuad: Mesh,
  overlapRenderTarget: WebGLRenderTarget,
  pathNumber: number
) {
  const n = frames.length;
  const totalPairs = (n * (n - 1)) / 2;
  let donePairs = 0;

  const result = Array.from({ length: n }, () => Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    result[i][i] = 1;
    for (let j = i + 1; j < n; j++) {
      // Pixels from A also seen by B
      result[i][j] = gpuOverlap(
        renderer,
        overlapScene,
        overlapCamera,
        overlapQuad,
        overlapRenderTarget,
        frames[i],
        frames[j]
      );

      // Pixels from B also seen by A
      result[j][i] = gpuOverlap(
        renderer,
        overlapScene,
        overlapCamera,
        overlapQuad,
        overlapRenderTarget,
        frames[j],
        frames[i]
      );

      donePairs++;

      // Progress update every 50 pairs
      if (donePairs % 50 === 0 || donePairs === totalPairs) {
        console.log(
          `${MESSAGE_TYPES.COVIZ_MATRIX_PROGRESS}_${JSON.stringify({
            pathNumber,
            progress: donePairs / totalPairs,
          })}`
        );
      }
    }
  }

  return result;
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

  const width = depthTarget.width;
  const height = depthTarget.height;

  const metadata = {
    pathNumber: currentPathNumber,
    frameIndex: currentFrameIndex,
    latitude: currentPoint.lat,
    longitude: currentPoint.lng,
    altitude: currentPoint.altitude,
    heading: currentPoint.heading,
    tilt: currentPoint.tilt,
    roll: currentPoint.roll,
    width: width,
    height: height,
    cameraFov: camera.fov,
    cameraNear: camera.near,
    cameraFar: camera.far,
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

  // Then render depth view and take screenshot
  // 1. Render scene to depth texture (depthTarget)
  renderer.setRenderTarget(depthTarget);
  renderer.render(scene, camera);

  // 2. Pack the depth texture into RGBA8 in a second pass
  renderer.setRenderTarget(packedDepthTarget);
  renderer.render(depthScene, depthCamera);

  // TODO: this step is the slowest
  const buffer = new Uint8Array(width * height * 4);
  renderer.readRenderTargetPixels(packedDepthTarget, 0, 0, width, height, buffer);

  const matrix = [];
  for (let y = height - 1; y >= 0; y--) {
    const row = new Array(width);
    for (let x = 0; x < width; x++) {
      const i = 4 * (y * width + x);
      const depthInMeters =
        decodeDepthFromGrayscaleRGBA(buffer[i], buffer[i + 1], buffer[i + 2], buffer[i + 3]) *
        camera.far;

      // NOTE: set roundDepthMatrix to 0 to disable rounding
      row[x] = Math.round(depthInMeters * 10 ** roundDepthMatrix) / 10 ** roundDepthMatrix;
    }
    matrix.push(row);
  }

  // -------------------------------------------------------------------
  // Store data required for overlap computation
  // -------------------------------------------------------------------
  const dataTexture = new DataTexture(buffer, width, height, RGBAFormat, UnsignedByteType);
  dataTexture.flipY = true;
  dataTexture.minFilter = LinearFilter;
  dataTexture.magFilter = LinearFilter;
  dataTexture.needsUpdate = true;

  currentPathFrames.push({
    dataTexture: dataTexture,
    camera: camera.clone(),
    width: width,
    height: height,
    cameraFar: camera.far,
  });

  // 3. Optional: render depth visualization quad to canvas
  // NOTE: We should avoid rendering in production to save some time
  if (debugDepthFrame) {
    renderer.setRenderTarget(null);
    renderer.render(depthScene, depthCamera);
  }

  console.log(
    `${MESSAGE_TYPES.DEPTH_FRAME_READY}_${JSON.stringify({
      pathNumber: currentPathNumber,
      frameIndex: currentFrameIndex,
      depthMatrix: matrix,
    })}`
  );

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
  }
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

  const width = Number(urlParams.get('width')) || DEFAULT_RENDERER_WIDTH;

  if (isNaN(width) || width <= 0) {
    throw new Error('Invalid width value');
  }

  const height = Number(urlParams.get('height')) || DEFAULT_RENDERER_HEIGHT;

  if (isNaN(height) || height <= 0) {
    throw new Error('Invalid height value');
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

  return {
    token,
    csvUrls,
    cameraFar,
    cameraNear,
    cameraFov,
    width,
    height,
    debugDepthFrame,
    roundDepthMatrix,
  };
}

async function main() {
  // Fetch all query parameters from URL
  const {
    token,
    csvUrls,
    cameraFar,
    cameraNear,
    cameraFov,
    width,
    height,
    debugDepthFrame,
    roundDepthMatrix,
  } = parseQueryParams();

  const scene = new Scene();
  const raycaster = new Raycaster();

  const renderer = new WebGLRenderer({ antialias: true });
  renderer.setClearColor(0x87ceeb);
  document.body.appendChild(renderer.domElement);

  const camera = new PerspectiveCamera(
    cameraFov,
    window.innerWidth / window.innerHeight,
    cameraNear,
    cameraFar
  );
  const tiles = new TilesRenderer();

  const { depthTarget, packedDepthTarget, depthScene, depthCamera } = setupDepthRendering(
    width,
    height,
    cameraNear,
    cameraFar
  );

  const { overlapScene, overlapCamera, overlapQuad, overlapRenderTarget } = setupOverlapResources(
    width,
    height
  );

  // TODO: how do we know that the provided token is valid at runtime?
  reinstantiateTiles(tiles, camera, renderer, scene, token);

  onWindowResize(width, height, camera, renderer, depthTarget, packedDepthTarget);
  window.addEventListener(
    'resize',
    () => onWindowResize(width, height, camera, renderer, depthTarget, packedDepthTarget),
    false
  );
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
    currentPathFrames = [];

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
        packedDepthTarget,
        depthCamera,
        depthScene,
        csvUrls.length,
        currentPathIndex,
        currentPathNumber,
        currentCsvUrl,
        currentPathData,
        0,
        0,
        debugDepthFrame,
        roundDepthMatrix
      );

      // Wait for the current path to complete before moving to the next one
      while (currentPathPending) {
        await new Promise((_resolve) => setTimeout(_resolve, PATH_WAIT_DELAY_IN_MS));
      }

      // No need to continue if the path is empty
      if (currentPathFrames.length === 0) {
        continue;
      }

      // -------------------------------------------------------------
      // Path finished – compute and emit coviz matrix
      // -------------------------------------------------------------
      const covizMatrix = computeCovizMatrix(
        currentPathFrames,
        renderer,
        overlapScene,
        overlapCamera,
        overlapQuad,
        overlapRenderTarget,
        currentPathNumber
      );

      console.log(
        `${MESSAGE_TYPES.COVIZ_MATRIX_READY}_${JSON.stringify({
          pathNumber: currentPathNumber,
          covizMatrix: covizMatrix,
        })}`
      );

      // Notify puppeteer via postMessage and wait for confirmation
      try {
        await new Promise((resolve) => {
          const handler = (event: MessageEvent) => {
            if (event.data === MESSAGE_TYPES.COVIZ_MATRIX_CAPTURED) {
              window.removeEventListener('message', handler);
              resolve(undefined);
            }
          };

          window.addEventListener('message', handler);
          window.postMessage(MESSAGE_TYPES.COVIZ_MATRIX_READY, '*');

          setTimeout(() => {
            window.removeEventListener('message', handler);
            console.warn('Coviz matrix capture confirmation timed out');
            resolve(undefined);
          }, MESSAGE_DELAY_IN_MS);
        });
      } catch (e) {
        console.error('Error waiting for coviz matrix capture confirmation:', e);
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
