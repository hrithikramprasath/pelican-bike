# Pelican Riding a Bicycle — Browser WebAssembly Edition

> **100% AI-Generated Project (Gemini 4 Hydrogen)**  
> This standalone GitHub Pages branch (`gh-pages`) runs the **Pelican Riding a Bicycle** simulation directly in your browser using Google DeepMind's official **`@mujoco/mujoco@4.0.1` WebAssembly** physics engine, **Three.js** PBR rendering, and the trained **PPO + LQR** controller exported to JSON—with zero Python or server backend required.
>
> Created to test the end-to-end engineering capabilities of **Gemini 4 Hydrogen** (representing a dramatic leap over the author's earlier **Gemini 3.8 Flash** test).

- **Live Demo:** https://hrithikramprasath.github.io/pelican-bike/
- **Native Python + RL Training + MuJoCo Studio Source (`main` branch):** https://github.com/hrithikramprasath/pelican-bike/tree/main

## Files on this Branch (`gh-pages`)

- `index.html`, `styles.css`, `app.js` — Interactive browser frontend & 60 FPS simulation loop
- `viewer.js` — Three.js PBR scene & MuJoCo geometry/texture synchronization
- `controller.js`, `policy.js`, `track.js` — Client-side port of the 50 Hz PPO MLP + LQR safety shield, phase-locked track features, and 6 stunt state machines (`Turbo Boost!`, `Wheelie!`, `Wing-Flap Hop!`, `Stoppie!`, `No-Hands!`, `S-Carve!`)
- `trained_policy.json` — Exported `22 -> 96 -> 96 -> 4` `tanh` actor weights & LQR gain table
- `scenes/*.xml`, `assets/pelican_bicycle/` — Portable MJCF scenes, Blender-sculpted `.obj` meshes, and PBR textures
