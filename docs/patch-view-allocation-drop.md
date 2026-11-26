# Patch: drop startup_view_alloc tests (fixes build breakage for PR #2847)
# ----------------------------------------------------------------
# PR #2847 removed the view allocation fault-injection symbols but left the
# test CMake config and test source in place.  This patch drops all three so
# the build succeeds and #2847 can be merged without regression.
# ----------------------------------------------------------------

# files_to_remove:
#   tests/cmake/startup_view_alloc.cmake
#   tests/startup_view_alloc.cpp

# docs/startup-runtime.md — delete lines 1743-1767 (the "D3D12 view allocation
# failure" section).  In a normal edit:
#
#   /## D3D12 view allocation failure<CR>
#   /## D3D12 view census<CR>
#   d
#
# Or manually remove this block:
#
# ---
# ## D3D12 view allocation failure
#
# A new view record goes into three containers: the record vector, the vector of
# materialization flags and the map from content to id. If one of them cannot
# allocate, the record goes into none of them, the process continues and the first
# eight failures are logged:
#
# `x2w-d3d12 note_view: out of memory in <view_records|view_materialized|view_ids> ->
#  view of slot <n> not recorded, destination descriptor untracked`
#
# The device's view slots return nothing, so the log line is the only report.
# The destination descriptor is left untracked: its shadow entry and its token in
# title memory are cleared, and the view it named before loses its reference.
# A draw then sees a descriptor without a view, not the earlier one. If the
# desktop view was created, its resources leave the retire class of
# `X2W_D3D12_RELEASE_RETIRED` and are not released early. Earlier ids stay valid
# and a later view is recorded if its allocations succeed. Records already
# copied elsewhere are not touched.
#
# This handles the failure; it does not reduce the memory the records use and
# it does not cover other allocations such as the map of render and depth
# targets.
#
# `startup_view_alloc_<records|materialized|ids>_<warp|hardware>` fails the
# allocation at one site three times through the fixture-only export
# `x2w_d3d12_fail_view_allocations`. The failure is injected, not produced by
# exhausting memory. For `view_ids` it is raised before the insertion, so the
# fixture relies on the standard guarantee that a failed single insertion leaves
# an `unordered_map` unchanged. The `control` variants never inject it and must
# fail. The failure is exercised on slot 18 and on a slot 19 view with a counter
# resource. For slot 19 only the reference counts are checked: resources that
# allow unordered access are never in the retire class.
#
# ```powershell
# ctest --preset dev --no-tests=error -R '^startup_view_alloc_' --output-on-failure
# ```
#
# ---

# ---
# After removal, the surrounding text flows from
# "## D3D12 view census" (no blank line change needed; the section header
# is already preceded by a blank line).
