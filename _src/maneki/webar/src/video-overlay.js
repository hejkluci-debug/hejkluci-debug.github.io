// Animated overlay from a "stacked alpha" video: the top half of each frame is
// the colour, the bottom half the alpha mask. Plays as plain H.264/VP9, so it
// works in iOS Safari and Android Chrome (no WebM-alpha / HEVC-alpha needed).
import { Mesh, NoColorSpace, PlaneGeometry, ShaderMaterial, Vector3, VideoTexture, DoubleSide } from 'three';

export function createVideoElement(base, sources) {
  const v = document.createElement('video');
  v.muted = true; v.loop = true; v.playsInline = true;
  v.setAttribute('playsinline', ''); v.setAttribute('muted', '');
  v.crossOrigin = 'anonymous'; v.preload = 'auto';
  for (const src of sources) {
    const s = document.createElement('source');
    s.src = `${base}/${src}`;
    s.type = src.endsWith('.mp4') ? 'video/mp4; codecs="avc1.640028"' : 'video/webm; codecs="vp9"';
    v.appendChild(s);
  }
  v.load();
  return v;
}

export function createVideoOverlay(scene, video) {
  const { affine: aff } = scene;
  const [x0, y0, x1, y1] = scene.video.rect;
  const tex = new VideoTexture(video);
  tex.colorSpace = NoColorSpace;          // raw sRGB values in, raw values out
  const mat = new ShaderMaterial({
    transparent: true, depthTest: false, depthWrite: false, side: DoubleSide,
    uniforms: { map: { value: tex }, tint: { value: new Vector3(1, 1, 1) }, opacity: { value: 0 } },
    vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
    fragmentShader: `uniform sampler2D map; uniform vec3 tint; uniform float opacity; varying vec2 vUv;
      void main(){
        vec3 c = texture2D(map, vec2(vUv.x, 0.5 + vUv.y * 0.5)).rgb;
        float a = texture2D(map, vec2(vUv.x, vUv.y * 0.5)).r;
        gl_FragColor = vec4(c * tint, a * opacity);
      }`,
  });
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  const mesh = new Mesh(new PlaneGeometry((x1 - x0) * aff.sx, (y1 - y0) * -aff.sy), mat);
  mesh.position.set(aff.sx * cx + aff.tx, aff.sy * cy + aff.ty, 0);
  mesh.renderOrder = 2;
  let fade = 0;
  return {
    group: mesh,
    reset() { fade = 0; mat.uniforms.opacity.value = 0; video.currentTime = 0; video.play().catch(() => {}); },
    update(t, dt) { fade = Math.min(1, fade + dt / 0.4); mat.uniforms.opacity.value = fade; },
    setTint(r, g, b) { mat.uniforms.tint.value.set(r, g, b); },
  };
}
