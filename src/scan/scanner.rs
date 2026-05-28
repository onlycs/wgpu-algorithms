use super::pipeline::ScanPipeline;
use crate::common;

struct ScanLevel {
    bg: wgpu::BindGroup,
    x_groups: u32,
    y_groups: u32,
}

pub struct Scanner {
    pipeline: ScanPipeline,
    buf_hist: wgpu::Buffer,
    #[allow(dead_code)]
    scratch_buffer: Option<wgpu::Buffer>,
    levels: Vec<ScanLevel>,
    // The level-0 aux buffer + offset. After record_scan, this contains the
    // fully-resolved per-scan-chunk prefix sums that scatter needs to add in.
    // None means there's only one scan-chunk (no add ever needed).
    level0_aux: Option<(wgpu::Buffer, u64)>,
}

impl Scanner {
    pub fn new(device: &wgpu::Device, hist_bytes: u64) -> Self {
        let pipeline = ScanPipeline::new(device);

        let buf_hist = common::buffers::create_empty_storage_buffer(
            device,
            "sort/scanner/buffer:hist",
            hist_bytes,
        );

        let num_items = (hist_bytes / 4) as u32;
        let needed_bytes = pipeline.get_scratch_size(num_items);
        let scratch_buffer = (needed_bytes > 0).then(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sort/scanner/buffer:scratch"),
                size: needed_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        });

        let items_per_block = pipeline.vt * pipeline.block_size;
        let max_dispatch = 65535u32;

        // (data_is_buf_hist, byte_offset, count)
        // Level 0's data lives in buf_hist (in-place scan, no copy needed).
        // All later levels live inside the scratch buffer.
        let mut temp: Vec<(bool, u64, u32)> = vec![(true, 0, num_items)];
        let mut scratch_offset = 0u64;
        let mut levels = Vec::new();
        let mut level0_aux: Option<(wgpu::Buffer, u64)> = None;

        loop {
            let (data_is_buf_hist, data_off, count) = *temp.last().unwrap();
            if count <= 1 {
                break;
            }

            let aux_count = (count + items_per_block - 1) / items_per_block;
            let aux_size = (aux_count * 4) as u64;
            let aux_offset = common::math::align_to(scratch_offset, 256);

            let scratch = scratch_buffer.as_ref().unwrap();
            let data_buf: &wgpu::Buffer = if data_is_buf_hist { &buf_hist } else { scratch };

            // Stash level 0's aux location for the sorter to consume.
            if levels.is_empty() {
                level0_aux = Some((scratch.clone(), aux_offset));
            }

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: data_buf,
                            offset: data_off,
                            size: None,
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: scratch,
                            offset: aux_offset,
                            size: None,
                        }),
                    },
                ],
            });

            let workgroups = common::math::calc_groups(count, items_per_block);
            levels.push(ScanLevel {
                bg,
                x_groups: workgroups.min(max_dispatch),
                y_groups: (workgroups + max_dispatch - 1) / max_dispatch,
            });

            temp.push((false, aux_offset, aux_count));
            scratch_offset = aux_offset + aux_size;
        }

        Self {
            pipeline,
            buf_hist,
            scratch_buffer,
            levels,
            level0_aux,
        }
    }

    pub fn buf_hist(&self) -> &wgpu::Buffer {
        &self.buf_hist
    }

    /// Number of items processed per scan workgroup. The sorter needs this
    /// to compute which scan-chunk each of its histogram entries lives in,
    /// so it can fetch the right aux value during the fused add+scatter.
    pub fn items_per_scan_block(&self) -> u32 {
        self.pipeline.vt * self.pipeline.block_size
    }

    /// Level-0 aux buffer + byte offset. After `record_scan` returns,
    /// this region holds the fully-resolved per-scan-chunk prefix sums
    /// that scatter needs to add in. Returns None if the histogram fits
    /// in a single scan-chunk (in which case no add is ever needed).
    pub fn level0_aux(&self) -> Option<(&wgpu::Buffer, u64)> {
        self.level0_aux.as_ref().map(|(b, o)| (b, *o))
    }

    pub fn record_scan(&self, cpass: &mut wgpu::ComputePass) {
        // Scan-down: each level reads its data in place and produces its aux.
        // No initial copy needed — level 0's data is buf_hist itself, which
        // the radix reduce kernel has just written.
        for level in &self.levels {
            cpass.set_pipeline(&self.pipeline.scan_pipeline);
            cpass.set_bind_group(0, &level.bg, &[]);
            cpass.dispatch_workgroups(level.x_groups, level.y_groups, 1);
        }

        // Add-up: propagate higher-level prefixes downward.
        // We stop *before* the bottom-most add — the radix scatter kernel
        // will do that final add itself as part of its histogram lookup,
        // saving one full read+write pass over the histogram per radix pass.
        if self.levels.len() > 1 {
            for level in self.levels.iter().rev().take(self.levels.len() - 1) {
                cpass.set_pipeline(&self.pipeline.add_pipeline);
                cpass.set_bind_group(0, &level.bg, &[]);
                cpass.dispatch_workgroups(level.x_groups, level.y_groups, 1);
            }
        }
    }
}
