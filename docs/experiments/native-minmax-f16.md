# Direct DXIL `v_min_f16` and `v_max_f16`

Lowers `v_min_f16` (VOP2 `0x3a` / VOP3A `0x13a`) and `v_max_f16` (VOP2 `0x39` /
VOP3A `0x139`) in the native direct-DXIL compute path
(`src/gfx1013_direct_compute.inl`) using exact 32-bit integer IEEE-754 binary16
ordering while preserving `D[31:16]`.

## Scope

- Covers `-0.0h` vs `+0.0h` in both operand orders, normals, subnormals,
  `±Inf`, single-NaN `minNum`/`maxNum`, dual-NaN canonical `0x7e00`, all nine
  inline float constants (`240..248`: `±0.5`, `±1.0`, `±2.0`, `±4.0`, and
  `1/(2*pi)` -> `0x3118`), VOP3A `abs`/`neg`/`clamp`, and `D[31:16]`
  preservation.
- Refuses unsupported modifiers (`omod`, `op_sel`, unused `src2`) and
  graphics-input mode.
- `native_minmax_f16_ir` evaluates the emitted IR on CPU against the Python
  reference model for both the main batch (32 results) and all 9 inline float
  constants (27 results), plus modifier refusals.
- Negative controls: ignoring signed zero, propagating single NaN, omitting
  `clamp`, treating inline constants as zero, and zeroing `D[31:16]` each
  differ from the reference.
- Does not establish full-game execution.

## Reproduction

```powershell
.\tools\dev.ps1
cmake --preset dev
cmake --build --preset dev --target native_compute_ir_dump compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_minmax_f16_' -V
```
