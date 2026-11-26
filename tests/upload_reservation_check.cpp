#include "startup_stubs/upload_reservation.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

void check(bool ok, const char* what)
{
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}

constexpr std::uint64_t kAlign = 512;

}  // namespace

int main()
{
    unsigned char storage_a[4 * kAlign];
    unsigned char storage_b[8 * kAlign];
    std::vector<UploadRingChunk> chunks;
    chunks.push_back({nullptr, storage_a, sizeof(storage_a), 0});

    const auto first = reserve_upload_chunk(chunks[0], 2 * kAlign);
    check(first.mapped == storage_a && first.offset == 0, "initial reservation");
    check(chunks[0].used == 2 * kAlign, "initial cursor");
    first.mapped[0] = 0xa5;
    first.mapped[2 * kAlign - 1] = 0x5a;

    const auto refused = reserve_upload_chunk(chunks[0], 3 * kAlign);
    check(!refused.mapped && !refused.resource && refused.offset == 0,
          "insufficient tail refuses the reservation");
    check(chunks[0].used == 2 * kAlign, "refusal preserves cursor");

    chunks.push_back({nullptr, storage_b, sizeof(storage_b), 0});
    const auto second = reserve_upload_chunk(chunks[1], 3 * kAlign);
    check(second.mapped == storage_b && second.offset == 0, "new chunk reservation");
    second.mapped[0] = 0xc3;
    chunks.reserve(chunks.capacity() + 1);
    check(first.mapped[0] == 0xa5 && first.mapped[2 * kAlign - 1] == 0x5a,
          "first mapped reservation survives vector growth");
    check(second.mapped[0] == 0xc3, "new mapped reservation survives vector growth");

    const auto next = reserve_upload_chunk(chunks[1], 5 * kAlign);
    check(next.mapped == storage_b + 3 * kAlign && next.offset == 3 * kAlign,
          "exact remaining active tail");
    check(chunks[1].used == chunks[1].capacity, "active cursor reaches capacity");
    check(!reserve_upload_chunk(chunks[1], kAlign).mapped, "full active chunk refuses");
    check(chunks[1].used == chunks[1].capacity, "active refusal preserves cursor");

    const auto older = reserve_upload_chunk(chunks[0], 2 * kAlign);
    check(older.mapped == storage_a + 2 * kAlign && older.offset == 2 * kAlign,
          "older chunk reserves only previously unused bytes");
    older.mapped[0] = 0x7d;
    check(chunks[0].used == chunks[0].capacity, "older cursor reaches capacity");
    check(!reserve_upload_chunk(chunks[0], kAlign).mapped, "full older chunk refuses");
    check(chunks[0].used == chunks[0].capacity && storage_a[0] == 0xa5 &&
          storage_a[2 * kAlign - 1] == 0x5a && storage_a[2 * kAlign] == 0x7d,
          "existing reservations survive exhaustion");

    reset_upload_chunk(chunks[0]);
    check(chunks[0].used == 0 && chunks[0].unsubmitted_lists == 0 && chunks[0].submits.empty(),
          "reset clears cursor, unsubmitted count and submits");
    const auto recycled = reserve_upload_chunk(chunks[0], kAlign);
    check(recycled.mapped == storage_a && recycled.offset == 0,
          "recycled chunk reserves from start");

    // Guard fence logic and unsubmitted tracking:
    struct MockFence {
        std::uint64_t completed = 0;
    };
    MockFence fence_a;
    MockFence fence_b;
    const auto fence_query = [](const ID3D12Fence* f) {
        return reinterpret_cast<const MockFence*>(f)->completed;
    };

    check(is_upload_chunk_idle(chunks[0], fence_query), "chunk without unsubmitted lists or submits is idle");

    chunks[0].unsubmitted_lists = 1;
    check(!is_upload_chunk_idle(chunks[0], fence_query), "chunk with unsubmitted lists is not idle");

    chunks[0].unsubmitted_lists = 0;
    chunks[0].submits.emplace_back(reinterpret_cast<ID3D12Fence*>(&fence_a), 10);
    fence_a.completed = 9;
    check(!is_upload_chunk_idle(chunks[0], fence_query), "chunk with in-flight submit is not idle");

    fence_a.completed = 10;
    check(is_upload_chunk_idle(chunks[0], fence_query), "chunk with completed submit is idle");

    // Multi-queue submits:
    chunks[0].submits.emplace_back(reinterpret_cast<ID3D12Fence*>(&fence_b), 5);
    fence_b.completed = 4;
    check(!is_upload_chunk_idle(chunks[0], fence_query), "chunk with second pending queue submit is not idle");

    fence_b.completed = 5;
    check(is_upload_chunk_idle(chunks[0], fence_query), "chunk with all queue submits completed is idle");

    reset_upload_chunk(chunks[0]);
    check(chunks[0].used == 0 && chunks[0].unsubmitted_lists == 0 && chunks[0].submits.empty(),
          "final reset clears submits");

    std::puts("upload_reservation_check: all checks passed");
    return 0;
}
