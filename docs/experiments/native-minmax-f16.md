# Direct DXIL `v_min_f16` and `v_max_f16`

Lowers `v_min_f16` (VOP2 `0x3a` / VOP3A `0x13a`) and `v_max_f16` (VOP2 `0x39` /
VOP3A `0x139`) in the native direct-DXIL compute path
(`src/gfx1013_direct_compute.inl`) using exact 32-bit integer IEEE-754 binary16
ordering while preserving `D[31:16]`.

## Scope

- Covers `-0.0h` vs `+0.0h` in both operand orders, normals, subnormals,
  `±Inf`, single-NaN `minNum`/`maxNum`, dual-NaN canonical `0x7e00`, VOP3A
  `abs`/`neg`/`clamp`, and `D[31:16]` preservation.
- Refuses unsupported modifiers (`omod`, `op_sel`, unused `src2`).
- Negative controls: ignoring signed zero, propagating single NaN, omitting
  `clamp`, and zeroing `D[31:16]` each differ from the reference.
- Does not establish full-game execution.

## Reproduction

```powershell
.\tools\dev.ps1
cmake --preset dev
cmake --build --preset dev --target compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_minmax_f16_' -V
```
