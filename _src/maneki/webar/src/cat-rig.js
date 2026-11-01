// Maneki-neko arm rig: a textured grid mesh whose vertices rotate around the
// shoulder pivot by angle * w, where w ramps 0 -> 1 along the arm axis
// (linear blend skinning with one bone). Same math as animation/preview.py.
import {
  BufferAttribute, BufferGeometry, DoubleSide, Group, Mesh, MeshBasicMaterial,
  PlaneGeometry, SRGBColorSpace, TextureLoader,
} from 'three';

const smooth = (t, a, b) => {
  const x = Math.min(Math.max((t - a) / (b - a), 0), 1);
  return x * x * (3 - 2 * x);
};

export async function loadTextures(base, names) {
  const loader = new TextureLoader();
  const webp = await supportsWebp();
  return Promise.all(names.map((n) => loader.loadAsync(`${base}/${n}.${webp ? 'webp' : 'png'}`)
    .then((t) => { t.colorSpace = SRGBColorSpace; return t; })));
}

function supportsWebp() {
  return new Promise((res) => {
    const img = new Image();
    img.onload = () => res(img.width === 1);
    img.onerror = () => res(false);
    img.src = 'data:image/webp;base64,UklGRhoAAABXRUJQVlA4TA0AAAAvAAAAEAcQERGIiP4HAA==';
  });
}

// rect = [x0, y0, x1, y1] in cat pixel space; returns a mesh in anchor space
function texturedQuad(rect, texture, aff, cols = 1, rows = 1, renderOrder = 0) {
  const [x0, y0, x1, y1] = rect;
  const geo = new PlaneGeometry(1, 1, cols, rows);
  const pos = geo.attributes.position;
  const pix = new Float32Array(pos.count * 2);       // rest pose in pixel space
  for (let i = 0; i < pos.count; i++) {
    const u = pos.getX(i) + 0.5, v = 0.5 - pos.getY(i); // 0..1, v down
    const px = x0 + u * (x1 - x0), py = y0 + v * (y1 - y0);
    pix[2 * i] = px; pix[2 * i + 1] = py;
    pos.setXYZ(i, aff.sx * px + aff.tx, aff.sy * py + aff.ty, 0);
  }
  pos.needsUpdate = true;
  const mat = new MeshBasicMaterial({ map: texture, transparent: true, side: DoubleSide, depthTest: false, depthWrite: false });
  const mesh = new Mesh(geo, mat);
  mesh.renderOrder = renderOrder;
  mesh.userData.pix = pix;
  return mesh;
}

export function createCatRig(scene, textures) {
  const { affine: aff, rig } = scene;
  const [armTex, underTex] = textures;
  const group = new Group();
  const under = texturedQuad(scene.underlay_rect, underTex, aff, 1, 1, 1);
  const arm = texturedQuad(scene.arm_rect, armTex, aff, scene.mesh.cols, scene.mesh.rows, 2);
  group.add(under, arm);

  const [px, py] = rig.pivot;
  let ax = rig.paw[0] - px, ay = rig.paw[1] - py;
  const n = Math.hypot(ax, ay); ax /= n; ay /= n;
  const pix = arm.userData.pix;
  const weights = new Float32Array(pix.length / 2);
  for (let i = 0; i < weights.length; i++) {
    const t = (pix[2 * i] - px) * ax + (pix[2 * i + 1] - py) * ay;
    weights[i] = smooth(t, rig.ramp[0], rig.ramp[1]);
  }
  const pos = arm.geometry.attributes.position;
  const { min_deg: lo, max_deg: hi, period_s: period } = rig.motion;
  let fade = 0;           // 0..1, eases the animation in after the target is found

  function setAngle(deg) {
    const th = deg * Math.PI / 180;
    for (let i = 0; i < weights.length; i++) {
      const a = th * weights[i], c = Math.cos(a), s = Math.sin(a);
      const dx = pix[2 * i] - px, dy = pix[2 * i + 1] - py;
      const x = px + c * dx - s * dy, y = py + s * dx + c * dy;
      pos.setXY(i, aff.sx * x + aff.tx, aff.sy * y + aff.ty);
    }
    pos.needsUpdate = true;
  }

  return {
    group,
    reset() { fade = 0; setAngle(0); },
    setTint(r, g, b) {
      for (const m of [under, arm]) m.material.color.setRGB(r, g, b, SRGBColorSpace);
    },
    update(tSec, dt) {
      fade = Math.min(1, fade + dt / 0.6);
      const s = 0.5 - 0.5 * Math.cos(2 * Math.PI * tSec / period);
      setAngle(fade * (lo + (hi - lo) * s));
    },
  };
}
