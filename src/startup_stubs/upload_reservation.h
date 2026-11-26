#pragma once

#include <cstdint>
#include <utility>
#include <vector>

struct ID3D12Resource;
struct ID3D12Fence;

struct UploadRingChunk {
    ID3D12Resource* resource = nullptr;
    unsigned char* mapped = nullptr;
    std::uint64_t capacity = 0;
    std::uint64_t used = 0;
    std::size_t unsubmitted_lists = 0;
    std::vector<std::pair<ID3D12Fence*, std::uint64_t>> submits{};
};

struct UploadReservation {
    unsigned char* mapped = nullptr;
    std::uint64_t offset = 0;
    ID3D12Resource* resource = nullptr;
};

// Caller holds the lock. A successful reservation owns previously unused bytes.
inline UploadReservation reserve_upload_chunk(UploadRingChunk& chunk, std::uint64_t rounded)
{
    if (chunk.used > chunk.capacity || rounded > chunk.capacity - chunk.used) return {};
    const auto start = chunk.used;
    chunk.used += rounded;
    return {chunk.mapped + start, start, chunk.resource};
}

template<typename FenceQuery>
inline bool is_upload_chunk_idle(const UploadRingChunk& chunk, FenceQuery&& query)
{
    if (chunk.unsubmitted_lists != 0) return false;
    for (const auto& submit : chunk.submits) {
        if (!submit.first) return false;
        if (query(submit.first) < submit.second) return false;
    }
    return true;
}

inline void reset_upload_chunk(UploadRingChunk& chunk)
{
    chunk.used = 0;
    chunk.submits.clear();
    chunk.unsubmitted_lists = 0;
}
