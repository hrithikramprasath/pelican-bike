// Three.js PBR renderer synced directly to @mujoco/mujoco 4.0.1 WASM state.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export class PelicanThreeViewer {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x95c8eb);
    this.scene.fog = new THREE.FogExp2(0xb8dbf2, 0.0065);

    const w = container.clientWidth || window.innerWidth;
    const h = container.clientHeight || window.innerHeight;
    this.camera = new THREE.PerspectiveCamera(45, w / h, 0.05, 320);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(-2.3, -2.3, 1.5);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(w, h);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.maxPolarAngle = Math.PI * 0.49;
    this.controls.minDistance = 0.8;
    this.controls.maxDistance = 40.0;
    this.controls.target.set(0, 0, 0.65);

    this._setupSkyAndLights();

    this.geomMeshes = [];
    this.disposables = [];
    this.cameraMode = 'orbit';
    this.followCamera = true;
    this.fixedCamId = -1;
    this.bikeBodyId = -1;
    this.prevTargetValid = false;
    this.prevBikePos = new THREE.Vector3();

    this._onResize = () => this.resize();
    window.addEventListener('resize', this._onResize);
  }

  _setupSkyAndLights() {
    // Procedural studio-coastal sky dome + PMREM environment reflection
    const skyGeo = new THREE.SphereGeometry(200, 32, 20);
    const skyMat = new THREE.ShaderMaterial({
      side: THREE.BackSide,
      depthWrite: false,
      uniforms: {
        topColor: { value: new THREE.Color(0x2e619e) },
        horizonColor: { value: new THREE.Color(0x9bd2f5) },
        bottomColor: { value: new THREE.Color(0xd2c4a6) },
      },
      vertexShader: `
        varying vec3 vWorldPos;
        void main() {
          vec4 wp = modelMatrix * vec4(position, 1.0);
          vWorldPos = wp.xyz;
          gl_Position = projectionMatrix * viewMatrix * wp;
        }
      `,
      fragmentShader: `
        uniform vec3 topColor;
        uniform vec3 horizonColor;
        uniform vec3 bottomColor;
        varying vec3 vWorldPos;
        void main() {
          float h = normalize(vWorldPos).z;
          vec3 col = h >= 0.0
            ? mix(horizonColor, topColor, pow(clamp(h, 0.0, 1.0), 0.55))
            : mix(horizonColor, bottomColor, clamp(-h * 4.0, 0.0, 1.0));
          gl_FragColor = vec4(col, 1.0);
        }
      `,
    });
    this.skyDome = new THREE.Mesh(skyGeo, skyMat);
    this.scene.add(this.skyDome);

    const pmrem = new THREE.PMREMGenerator(this.renderer);
    const envScene = new THREE.Scene();
    envScene.add(new THREE.Mesh(new THREE.SphereGeometry(10, 16, 12), skyMat.clone()));
    const envRT = pmrem.fromScene(envScene, 0.04);
    this.scene.environment = envRT.texture;
    pmrem.dispose();

    const hemi = new THREE.HemisphereLight(0xbce0ff, 0xcab892, 0.72);
    this.scene.add(hemi);

    this.sunLight = new THREE.DirectionalLight(0xfff8ea, 2.25);
    this.sunLight.position.set(5, -6, 13);
    this.sunLight.castShadow = true;
    this.sunLight.shadow.mapSize.set(2048, 2048);
    this.sunLight.shadow.camera.near = 1.0;
    this.sunLight.shadow.camera.far = 42.0;
    const ext = 7.5;
    this.sunLight.shadow.camera.left = -ext;
    this.sunLight.shadow.camera.right = ext;
    this.sunLight.shadow.camera.top = ext;
    this.sunLight.shadow.camera.bottom = -ext;
    this.sunLight.shadow.bias = -0.0004;
    this.sunLight.shadow.normalBias = 0.015;
    this.scene.add(this.sunLight);
    this.scene.add(this.sunLight.target);

    const fill = new THREE.DirectionalLight(0x88aee0, 0.72);
    fill.position.set(-7, 7, 9);
    this.scene.add(fill);
  }

  _clearSceneModel() {
    for (const { mesh } of this.geomMeshes) {
      this.scene.remove(mesh);
    }
    this.geomMeshes = [];
    for (const d of this.disposables) {
      d.dispose();
    }
    this.disposables = [];
    this.prevTargetValid = false;
  }

  _makeDataTexture(model, texId, isSRGB, repeatX = 1, repeatY = 1) {
    if (texId < 0 || texId >= model.ntex) return null;
    const w = model.tex_width[texId];
    const h = model.tex_height[texId];
    const ch = model.tex_nchannel[texId];
    const adr = model.tex_adr[texId];
    const src = model.tex_data;
    const rgba = new Uint8Array(w * h * 4);
    if (ch === 4) {
      rgba.set(src.subarray(adr, adr + w * h * 4));
    } else if (ch === 3) {
      for (let i = 0, s = adr, d = 0; i < w * h; i++, s += 3, d += 4) {
        rgba[d] = src[s];
        rgba[d + 1] = src[s + 1];
        rgba[d + 2] = src[s + 2];
        rgba[d + 3] = 255;
      }
    } else {
      for (let i = 0, s = adr, d = 0; i < w * h; i++, s += ch, d += 4) {
        const v = src[s];
        rgba[d] = v;
        rgba[d + 1] = v;
        rgba[d + 2] = v;
        rgba[d + 3] = 255;
      }
    }
    const tex = new THREE.DataTexture(rgba, w, h, THREE.RGBAFormat);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(repeatX, repeatY);
    tex.anisotropy = 4;
    tex.minFilter = THREE.LinearMipmapLinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.generateMipmaps = true;
    tex.colorSpace = isSRGB ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    tex.needsUpdate = true;
    this.disposables.push(tex);
    return tex;
  }

  _buildMeshGeometry(model, meshId) {
    const fAdr = model.mesh_faceadr[meshId];
    const fNum = model.mesh_facenum[meshId];
    const vAdr = model.mesh_vertadr[meshId];
    const nAdr = model.mesh_normaladr[meshId];
    const tAdr = model.mesh_texcoordadr[meshId];
    const hasUV = model.mesh_texcoordnum[meshId] > 0;

    const verts = model.mesh_vert;
    const norms = model.mesh_normal;
    const texcoords = model.mesh_texcoord;
    const faces = model.mesh_face;
    const faceNorms = model.mesh_facenormal;
    const faceUVs = model.mesh_facetexcoord;

    const posArr = new Float32Array(fNum * 9);
    const normArr = new Float32Array(fNum * 9);
    const uvArr = hasUV ? new Float32Array(fNum * 6) : null;

    for (let f = 0; f < fNum; f++) {
      const fBase = (fAdr + f) * 3;
      for (let c = 0; c < 3; c++) {
        const vi = (vAdr + faces[fBase + c]) * 3;
        const ni = (nAdr + faceNorms[fBase + c]) * 3;
        const outV = f * 9 + c * 3;
        posArr[outV + 0] = verts[vi + 0];
        posArr[outV + 1] = verts[vi + 1];
        posArr[outV + 2] = verts[vi + 2];
        normArr[outV + 0] = norms[ni + 0];
        normArr[outV + 1] = norms[ni + 1];
        normArr[outV + 2] = norms[ni + 2];
        if (hasUV) {
          const ti = (tAdr + faceUVs[fBase + c]) * 2;
          const outT = f * 6 + c * 2;
          uvArr[outT + 0] = texcoords[ti + 0];
          uvArr[outT + 1] = texcoords[ti + 1];
        }
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(normArr, 3));
    if (hasUV) {
      geo.setAttribute('uv', new THREE.BufferAttribute(uvArr, 2));
    }
    this.disposables.push(geo);
    return geo;
  }

  buildSceneFromModel(mujoco, model, data) {
    this._clearSceneModel();
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;

    const G = mujoco.mjtGeom;
    const R = mujoco.mjtTextureRole;
    const texRolesPerMat = model.nmat > 0 ? model.mat_texid.length / model.nmat : 10;

    // Cache Three.js MeshStandardMaterial per MuJoCo material ID (non-ground)
    const matCache = new Map();
    const getMaterial = (matId, rgba, isPlane = false) => {
      const key = `${matId}_${rgba.map((x) => x.toFixed(3)).join(',')}_${isPlane}`;
      if (matCache.has(key)) return matCache.get(key);

      let color = new THREE.Color().setRGB(rgba[0], rgba[1], rgba[2], THREE.SRGBColorSpace);
      let roughness = 0.55;
      let metalness = 0.05;
      let map = null;
      let roughnessMap = null;
      let metalnessMap = null;
      let normalMap = null;
      let opacity = rgba[3];

      if (matId >= 0) {
        const mRgba = model.mat_rgba;
        color.setRGB(
          mRgba[matId * 4 + 0],
          mRgba[matId * 4 + 1],
          mRgba[matId * 4 + 2],
          THREE.SRGBColorSpace
        );
        opacity = mRgba[matId * 4 + 3];
        roughness = model.mat_roughness[matId] >= 0 ? model.mat_roughness[matId] : 0.65;
        metalness = model.mat_metallic[matId] >= 0 ? model.mat_metallic[matId] : 0.0;

        const base = matId * texRolesPerMat;
        const rgbTex = model.mat_texid[base + R.mjTEXROLE_RGB.value];
        const rghTex = model.mat_texid[base + R.mjTEXROLE_ROUGHNESS.value];
        const metTex = model.mat_texid[base + R.mjTEXROLE_METALLIC.value];
        const nrmTex = model.mat_texid[base + R.mjTEXROLE_NORMAL.value];

        const repX = isPlane ? 220 : 1;
        const repY = isPlane ? 220 : 1;
        map = this._makeDataTexture(model, rgbTex, true, repX, repY);
        roughnessMap = this._makeDataTexture(model, rghTex, false, repX, repY);
        metalnessMap = this._makeDataTexture(model, metTex, false, repX, repY);
        normalMap = this._makeDataTexture(model, nrmTex, false, repX, repY);
      }

      const mat = new THREE.MeshStandardMaterial({
        color,
        roughness: roughnessMap ? 1.0 : roughness,
        metalness: metalnessMap ? 1.0 : metalness,
        map,
        roughnessMap,
        metalnessMap,
        normalMap,
        transparent: opacity < 0.99,
        opacity,
      });
      if (isPlane) {
        mat.roughness = 0.92;
        mat.metalness = 0.0;
      }
      this.disposables.push(mat);
      matCache.set(key, mat);
      return mat;
    };

    const meshGeoCache = new Map();
    for (let g = 0; g < model.ngeom; g++) {
      const grp = model.geom_group[g];
      // Show environment/road (group 1) and high-detail PBR meshes (group 2); hide collision primitives (group 3)
      if (grp === 3) continue;

      const gtype = model.geom_type[g];
      const matId = model.geom_matid[g];
      const rgba = [
        model.geom_rgba[g * 4 + 0],
        model.geom_rgba[g * 4 + 1],
        model.geom_rgba[g * 4 + 2],
        model.geom_rgba[g * 4 + 3],
      ];
      const sx = model.geom_size[g * 3 + 0];
      const sy = model.geom_size[g * 3 + 1];
      const sz = model.geom_size[g * 3 + 2];

      let geo = null;
      const isPlane = gtype === G.mjGEOM_PLANE.value;
      if (isPlane) {
        geo = new THREE.PlaneGeometry(240, 240);
        this.disposables.push(geo);
      } else if (gtype === G.mjGEOM_BOX.value) {
        geo = new THREE.BoxGeometry(sx * 2, sy * 2, sz * 2);
        this.disposables.push(geo);
      } else if (gtype === G.mjGEOM_SPHERE.value) {
        geo = new THREE.SphereGeometry(sx, 24, 16);
        this.disposables.push(geo);
      } else if (gtype === G.mjGEOM_CYLINDER.value) {
        geo = new THREE.CylinderGeometry(sx, sx, sy * 2, 24);
        geo.rotateX(Math.PI * 0.5);
        this.disposables.push(geo);
      } else if (gtype === G.mjGEOM_CAPSULE.value) {
        geo = new THREE.CapsuleGeometry(sx, sy * 2, 10, 20);
        geo.rotateX(Math.PI * 0.5);
        this.disposables.push(geo);
      } else if (gtype === G.mjGEOM_MESH.value) {
        const mid = model.geom_dataid[g];
        if (!meshGeoCache.has(mid)) {
          meshGeoCache.set(mid, this._buildMeshGeometry(model, mid));
        }
        geo = meshGeoCache.get(mid);
      }
      if (!geo) continue;

      const material = getMaterial(matId, rgba, isPlane);
      const mesh = new THREE.Mesh(geo, material);
      mesh.matrixAutoUpdate = false;
      mesh.castShadow = !isPlane && model.geom_castshadow[g] !== 0 && gtype !== G.mjGEOM_MESH.value ? true : (!isPlane && model.geom_dataid[g] !== 0);
      mesh.receiveShadow = true;
      this.scene.add(mesh);
      this.geomMeshes.push({ geomId: g, mesh });
    }

    this.bikeBodyId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, 'pb_pelican_bicycle');
    this.syncTransforms();
    this.resetOrbitCamera();
  }

  setCameraView(viewName) {
    this.cameraMode = viewName;
    if (viewName === 'orbit') {
      this.fixedCamId = -1;
      this.camera.matrixAutoUpdate = true;
      this.camera.fov = 45;
      this.camera.updateProjectionMatrix();
      this.controls.enabled = true;
      this.resetOrbitCamera();
      return;
    }
    const camId = this.mujoco.mj_name2id(
      this.model,
      this.mujoco.mjtObj.mjOBJ_CAMERA.value,
      viewName
    );
    if (camId >= 0) {
      this.fixedCamId = camId;
      this.controls.enabled = false;
      this.camera.fov = this.model.cam_fovy[camId] || 48;
      this.camera.updateProjectionMatrix();
    }
  }

  resetOrbitCamera() {
    if (!this.data || this.bikeBodyId < 0) return;
    const b3 = this.bikeBodyId * 3;
    const px = this.data.xpos[b3 + 0];
    const py = this.data.xpos[b3 + 1];
    const pz = this.data.xpos[b3 + 2];
    this.controls.target.set(px, py, pz + 0.65);
    this.camera.position.set(px - 2.35, py - 2.35, pz + 1.55);
    this.controls.update();
    this.prevBikePos.set(px, py, pz);
    this.prevTargetValid = true;
  }

  syncTransforms() {
    if (!this.data) return;
    const xpos = this.data.geom_xpos;
    const xmat = this.data.geom_xmat;

    for (let i = 0; i < this.geomMeshes.length; i++) {
      const { geomId, mesh } = this.geomMeshes[i];
      const p = geomId * 3;
      const m = geomId * 9;
      mesh.matrix.set(
        xmat[m + 0], xmat[m + 1], xmat[m + 2], xpos[p + 0],
        xmat[m + 3], xmat[m + 4], xmat[m + 5], xpos[p + 1],
        xmat[m + 6], xmat[m + 7], xmat[m + 8], xpos[p + 2],
        0, 0, 0, 1
      );
    }

    if (this.bikeBodyId >= 0) {
      const b3 = this.bikeBodyId * 3;
      const px = this.data.xpos[b3 + 0];
      const py = this.data.xpos[b3 + 1];
      const pz = this.data.xpos[b3 + 2];

      this.sunLight.position.set(px + 5.0, py - 6.0, 13.0);
      this.sunLight.target.position.set(px, py, 0.0);
      this.sunLight.target.updateMatrixWorld();
      this.skyDome.position.set(px, py, 0.0);

      if (this.fixedCamId >= 0) {
        const c3 = this.fixedCamId * 3;
        const c9 = this.fixedCamId * 9;
        const cp = this.data.cam_xpos;
        const cm = this.data.cam_xmat;
        this.camera.matrixAutoUpdate = false;
        this.camera.matrix.set(
          cm[c9 + 0], cm[c9 + 1], cm[c9 + 2], cp[c3 + 0],
          cm[c9 + 3], cm[c9 + 4], cm[c9 + 5], cp[c3 + 1],
          cm[c9 + 6], cm[c9 + 7], cm[c9 + 8], cp[c3 + 2],
          0, 0, 0, 1
        );
        this.camera.matrixWorld.copy(this.camera.matrix);
        this.camera.matrixWorldInverse.copy(this.camera.matrixWorld).invert();
      } else {
        if (this.followCamera && this.prevTargetValid) {
          const dx = px - this.prevBikePos.x;
          const dy = py - this.prevBikePos.y;
          const dz = pz - this.prevBikePos.z;
          this.controls.target.x += dx;
          this.controls.target.y += dy;
          this.controls.target.z += dz;
          this.camera.position.x += dx;
          this.camera.position.y += dy;
          this.camera.position.z += dz;
        }
        this.prevBikePos.set(px, py, pz);
        this.prevTargetValid = true;
        this.controls.update();
      }
    }
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }

  resize() {
    const w = this.container.clientWidth || window.innerWidth;
    const h = this.container.clientHeight || window.innerHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }
}
