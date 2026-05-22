use crate::common;
use crate::scan::Scanner;
use crate::sort::pipeline::SortPipeline;

pub struct Sorter {
    scanner: Scanner,
    pipeline: SortPipeline,
    num_elements: u32,
    buf_values: wgpu::Buffer,
    #[allow(dead_code)]
    buf_values_scratch: wgpu::Buffer,
    buf_keys: wgpu::Buffer,
    #[allow(dead_code)]
    buf_keys_scratch: wgpu::Buffer,
    uniform_buffers: Vec<wgpu::Buffer>,
    bind_groups: Vec<(wgpu::BindGroup, wgpu::BindGroup)>,
    // Buffer bound at the scatter's "aux" slot. When the histogram fits in
    // one scan-chunk there's no aux to add, so we bind a 4-byte dummy.
    #[allow(dead_code)]
    dummy_aux: wgpu::Buffer,
}

impl Sorter {
    pub fn new(device: &wgpu::Device, num_elements: u32) -> Self {
        let pipeline = SortPipeline::new(device);

        let capacity = num_elements as u64 * 4;
        let items_per_block = (pipeline.vt * pipeline.block_size) as u64;
        let num_blocks = (num_elements as u64 + items_per_block - 1) / items_per_block;
        let hist_bytes_aligned = common::math::align_to(num_blocks * 16, 256);

        let scanner = Scanner::new(device, hist_bytes_aligned);

        let buf_values = common::buffers::create_empty_storage_buffer(
            device,
            "sort/sorter/buffer:values",
            capacity,
        );
        let buf_values_scratch = common::buffers::create_empty_storage_buffer(
            device,
            "sort/sorter/buffer:values_scratch",
            capacity,
        );
        let buf_keys = common::buffers::create_empty_storage_buffer(
            device,
            "sort/sorter/buffer:keys",
            capacity,
        );
        let buf_keys_scratch = common::buffers::create_empty_storage_buffer(
            device,
            "sort/sorter/buffer:keys_scratch",
            capacity,
        );

        // Dummy aux for the single-scan-chunk case. Bound but never effectively
        // read (the shader's chunk-index calculation returns 0, which gates
        // the read).
        let dummy_aux = common::buffers::create_empty_storage_buffer(
            device,
            "sort/sorter/buffer:dummy_aux",
            16,
        );

        // Resolve the aux binding once. If the scanner has level-0 aux, use it;
        // otherwise fall back to the dummy buffer.
        let (aux_buffer, aux_offset): (&wgpu::Buffer, u64) = match scanner.level0_aux() {
            Some((b, off)) => (b, off),
            None => (&dummy_aux, 0),
        };

        let mut uniform_buffers = Vec::with_capacity(16);
        for i in 0..16 {
            uniform_buffers.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("sort/sorter/buffer:uniform{i}")),
                // Now holds: bit_index, num_items, num_blocks, items_per_scan_block
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }

        let mut bind_groups = Vec::with_capacity(16);
        for i in 0..16 {
            let (values_src, values_dst, keys_src, keys_dst) = if i % 2 == 0 {
                (&buf_values, &buf_values_scratch, &buf_keys, &buf_keys_scratch)
            } else {
                (&buf_values_scratch, &buf_values, &buf_keys_scratch, &buf_keys)
            };

            // Reduce: same layout as before. The scatter layout adds binding 6
            // for the aux buffer, but since both kernels share the bind group
            // layout, reduce also gets the (unused) binding.
            let reduce_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: values_src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: scanner.buf_hist().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: values_dst.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: uniform_buffers[i].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: keys_src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: keys_dst.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: aux_buffer,
                            offset: aux_offset,
                            size: None,
                        }),
                    },
                ],
            });

            let scatter_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: values_src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        // Scatter now reads buf_hist directly (in-place scan).
                        resource: scanner.buf_hist().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: values_dst.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: uniform_buffers[i].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: keys_src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: keys_dst.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: aux_buffer,
                            offset: aux_offset,
                            size: None,
                        }),
                    },
                ],
            });

            bind_groups.push((reduce_bg, scatter_bg));
        }

        Self {
            scanner,
            pipeline,
            num_elements,
            buf_values,
            buf_values_scratch,
            buf_keys,
            buf_keys_scratch,
            uniform_buffers,
            bind_groups,
            dummy_aux,
        }
    }

    pub fn buffer_values(&self) -> &wgpu::Buffer {
        &self.buf_values
    }

    pub fn buffer_keys(&self) -> &wgpu::Buffer {
        &self.buf_keys
    }

    pub fn sort(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        sort_length: u32,
    ) {
        assert!(
            sort_length <= self.num_elements,
            "sort_length ({sort_length}) exceeds buffer capacity ({})",
            self.num_elements
        );

        let n = sort_length as u64;
        let items_per_block = (self.pipeline.vt * self.pipeline.block_size) as u64;
        let num_blocks = (n + items_per_block - 1) / items_per_block;
        let items_per_scan_block = self.scanner.items_per_scan_block();

        for i in 0..16usize {
            let uniform_data = [
                (i * 2) as u32,
                sort_length,
                num_blocks as u32,
                items_per_scan_block,
            ];
            queue.write_buffer(
                &self.uniform_buffers[i],
                0,
                bytemuck::cast_slice(&uniform_data),
            );
        }

        let max_dispatch = 65535u32;
        let x_groups = (num_blocks as u32).min(max_dispatch);
        let y_groups = (num_blocks as u32 + max_dispatch - 1) / max_dispatch;

        let scanner = &self.scanner;
        let pipeline = &self.pipeline;
        let bind_groups = &self.bind_groups;

        for i in 0..16usize {
            let (reduce_bg, scatter_bg) = &bind_groups[i];

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                cpass.set_pipeline(&pipeline.reduce_pipeline);
                cpass.set_bind_group(0, reduce_bg, &[]);
                cpass.dispatch_workgroups(x_groups, y_groups, 1);
            }

            scanner.record_scan(encoder);

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                cpass.set_pipeline(&pipeline.scatter_pipeline);
                cpass.set_bind_group(0, scatter_bg, &[]);
                cpass.dispatch_workgroups(x_groups, y_groups, 1);
            }
        }
    }

    pub async fn sort_array(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        values: &[u32],
        keys: &[u32],
    ) -> (Vec<u32>, Vec<u32>) {
        let n = values.len() as u32;
        assert!(n <= self.num_elements);
        assert_eq!(values.len(), keys.len(), "values and keys must have the same length");

        queue.write_buffer(&self.buf_values, 0, bytemuck::cast_slice(values));
        queue.write_buffer(&self.buf_keys, 0, bytemuck::cast_slice(keys));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sort/sorter/encoder"),
        });
        self.sort(&mut encoder, queue, n);
        queue.submit(Some(encoder.finish()));

        let sorted_values =
            common::buffers::download_buffer(device, queue, &self.buf_values, n as u64 * 4).await;
        let sorted_keys =
            common::buffers::download_buffer(device, queue, &self.buf_keys, n as u64 * 4).await;

        (sorted_values, sorted_keys)
    }
}
