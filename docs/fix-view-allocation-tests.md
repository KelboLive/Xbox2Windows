# Fix for PR #2847: remove startup_view_alloc tests

PR #2847 removed these symbols but left the tests that depend on them:
  · x2w_d3d12_fail_view_allocations()
  · x2w_d3d12_view_probe() + ViewStoreProbe
  · view_allocations_to_fail[], view_allocation_gate()

The result is link errors when building the project.

## What to do (remove these three references)

1. **Delete the test source:**
   ```
   rm tests/startup_view_alloc.cpp
   ```

2. **Delete the CMake config:**
   ```
   rm tests/cmake/startup_view_alloc.cmake
   ```

3. **Remove the "D3D12 view allocation failure" documentation section
   from docs/startup-runtime.md (lines 1733–1767):**
   ```
   ## D3D12 view allocation failure
   ...
   ctest --preset dev --no-tests=error -R '^startup_view_alloc_' --output-on-failure
   ```

   In VS Code or any editor, delete lines 1733 through 1767 (inclusive).
   The section header "## D3D12 view census" on line 1768 should remain
   as the next heading after "## D3D12 resource budget".

## Verification (once you have a C++ build environment)

After applying the above, confirm the cmake configure step no longer
mentions `startup_view_alloc`:
   ```
   cmake --preset dev -N 2>&1 | grep view_alloc
   ```
   (expect no output)

Then build to verify no unresolved symbols:
   ```
   cmake --build --preset dev 2>&1 | grep -i failure
   ```
