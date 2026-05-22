@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> histograms: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;
@group(0) @binding(4) var<storage, read> keys_input: array<u32>;
@group(0) @binding(5) var<storage, read_write> keys_output: array<u32>;
// Level-0 aux from the scanner: fully-resolved prefix sums per scan-chunk.
// For a histogram entry at index p, the value to add is aux[chunk-1] where
// chunk = p / items_per_scan_block, with the convention that chunk 0 adds 0.
@group(0) @binding(6) var<storage, read> scan_aux: array<u32>;

struct Uniforms {
    bit_index: u32,
    num_items: u32,
    num_blocks: u32,
    items_per_scan_block: u32,
}

const VT: u32 = {{VT}}u;
const BLOCK_SIZE: u32 = {{BLOCK_SIZE}}u;
const ITEMS_PER_BLOCK: u32 = VT * BLOCK_SIZE;

var<workgroup> s_local_hist: array<vec4<u32>, {{BLOCK_SIZE}}>;

var<workgroup> s_counts: array<vec4<u32>, {{BLOCK_SIZE}}>;
var<workgroup> s_vals: array<u32, {{VT}} * {{BLOCK_SIZE}}>;
var<workgroup> s_keys: array<u32, {{VT}} * {{BLOCK_SIZE}}>;
var<workgroup> s_bucket_start: array<u32, 4>;

fn get_flat_group_id(group_id: vec3<u32>) -> u32 {
    return group_id.y * 65535u + group_id.x;
}

fn vec4_index(v: vec4<u32>, i: u32) -> u32 {
    if i == 0u { return v.x; }
    if i == 1u { return v.y; }
    if i == 2u { return v.z; }
    return v.w;
}

fn hist_lookup(idx: u32) -> u32 {
    let chunk = idx / uniforms.items_per_scan_block;
    var v = histograms[idx];
    if chunk > 0u {
        v += scan_aux[chunk - 1u];
    }
    return v;
}

@compute @workgroup_size(BLOCK_SIZE)
fn main_reduce(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tid = local_id.x;
    let flat_group_id = get_flat_group_id(group_id);
    let group_base_idx = flat_group_id * ITEMS_PER_BLOCK;
    let thread_base_idx = group_base_idx + (tid * VT);

    var counts: array<u32, 4>;

    if group_base_idx + ITEMS_PER_BLOCK <= uniforms.num_items {
        for (var i = 0u; i < VT; i++) {
            let val = input[thread_base_idx + i];
            let digit = (val >> uniforms.bit_index) & 3u;
            counts[digit] += 1u;
        }
    } else {
        for (var i = 0u; i < VT; i++) {
            let idx = thread_base_idx + i;
            if idx < uniforms.num_items {
                let val = input[idx];
                let digit = (val >> uniforms.bit_index) & 3u;
                counts[digit] += 1u;
            }
        }
    }

    s_local_hist[tid] = vec4<u32>(counts[0], counts[1], counts[2], counts[3]);
    workgroupBarrier();

    for (var s = (BLOCK_SIZE >> 1u); s > 0u; s >>= 1u) {
        if tid < s {
            s_local_hist[tid] += s_local_hist[tid + s];
        }
        workgroupBarrier();
    }

    if tid == 0u {
        let total_blocks = uniforms.num_blocks;
        let totals = s_local_hist[0];
        histograms[flat_group_id] = totals.x;
        histograms[total_blocks + flat_group_id] = totals.y;
        histograms[2u * total_blocks + flat_group_id] = totals.z;
        histograms[3u * total_blocks + flat_group_id] = totals.w;
    }
}

@compute @workgroup_size(BLOCK_SIZE)
fn main_scatter(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tid = local_id.x;
    let block_idx = get_flat_group_id(group_id);
    let group_base_idx = block_idx * ITEMS_PER_BLOCK;
    let thread_base_idx = group_base_idx + (tid * VT);
    let total_blocks = uniforms.num_blocks;
    let full_block = group_base_idx + ITEMS_PER_BLOCK <= uniforms.num_items;

    var my_vals: array<u32, {{VT}}>;
    var my_keys: array<u32, {{VT}}>;
    var my_digits: array<u32, {{VT}}>;
    var counts: array<u32, 4>;

    if full_block {
        for (var i = 0u; i < VT; i++) {
            let idx = thread_base_idx + i;
            let val = input[idx];
            let digit = (val >> uniforms.bit_index) & 3u;
            my_vals[i] = val;
            my_keys[i] = keys_input[idx];
            my_digits[i] = digit;
            counts[digit] += 1u;
        }
    } else {
        for (var i = 0u; i < VT; i++) {
            let idx = thread_base_idx + i;
            my_digits[i] = 4u;
            if idx < uniforms.num_items {
                let val = input[idx];
                let digit = (val >> uniforms.bit_index) & 3u;
                my_vals[i] = val;
                my_keys[i] = keys_input[idx];
                my_digits[i] = digit;
                counts[digit] += 1u;
            }
        }
    }

    s_counts[tid] = vec4<u32>(counts[0], counts[1], counts[2], counts[3]);
    workgroupBarrier();

    for (var offset = 1u; offset < BLOCK_SIZE; offset <<= 1u) {
        var temp = vec4<u32>(0u);
        if tid >= offset { temp = s_counts[tid - offset]; }
        workgroupBarrier();
        if tid >= offset { s_counts[tid] += temp; }
        workgroupBarrier();
    }

    let thread_inclusive = s_counts[tid];
    let my_counts_v = vec4<u32>(counts[0], counts[1], counts[2], counts[3]);
    let thread_excl = thread_inclusive - my_counts_v;

    if tid == 0u {
        let totals = s_counts[BLOCK_SIZE - 1u];
        s_bucket_start[0] = 0u;
        s_bucket_start[1] = totals.x;
        s_bucket_start[2] = totals.x + totals.y;
        s_bucket_start[3] = totals.x + totals.y + totals.z;
    }
    workgroupBarrier();

    var local_run: array<u32, 4>;
    for (var i = 0u; i < VT; i++) {
        let d = my_digits[i];
        if d < 4u {
            let excl = vec4_index(thread_excl, d);
            let local_dst = s_bucket_start[d] + excl + local_run[d];
            s_vals[local_dst] = my_vals[i];
            s_keys[local_dst] = my_keys[i];
            local_run[d] += 1u;
        }
    }
    workgroupBarrier();

    var base: array<u32, 4>;
    if block_idx > 0u {
        base[0] = hist_lookup(block_idx - 1u);
        base[1] = hist_lookup(total_blocks + block_idx - 1u);
        base[2] = hist_lookup(2u * total_blocks + block_idx - 1u);
        base[3] = hist_lookup(3u * total_blocks + block_idx - 1u);
    } else {
        base[0] = 0u;
        base[1] = hist_lookup(total_blocks - 1u);
        base[2] = hist_lookup(2u * total_blocks - 1u);
        base[3] = hist_lookup(3u * total_blocks - 1u);
    }

    let items_in_block = select(
        uniforms.num_items - group_base_idx,
        ITEMS_PER_BLOCK,
        full_block
    );

    let end0 = s_bucket_start[1];
    let end1 = s_bucket_start[2];
    let end2 = s_bucket_start[3];

    for (var i = tid; i < items_in_block; i += BLOCK_SIZE) {
        var d: u32;
        if i < end0 { d = 0u; }
        else if i < end1 { d = 1u; }
        else if i < end2 { d = 2u; }
        else { d = 3u; }

        let local_in_bucket = i - s_bucket_start[d];
        let dest = base[d] + local_in_bucket;
        output[dest] = s_vals[i];
        keys_output[dest] = s_keys[i];
    }
}
