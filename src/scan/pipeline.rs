use crate::common;

pub struct ScanPipeline {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub scan_pipeline: wgpu::ComputePipeline,
    pub add_pipeline: wgpu::ComputePipeline,
    pub vt: u32,
    pub block_size: u32,
}

impl ScanPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sort/scanner/bindgroup_layout"),
            entries: &[
                common::buffers::bind_entry(0, false, false),
                common::buffers::bind_entry(1, false, false),
            ],
        });

        let limits = device.limits();
        let max_shared_mem = limits.max_compute_workgroup_storage_size;

        // High End (M3/Desktop): 32KB+ shared mem -> Use VT=8, Block=256
        // Low End (Mobile): <32KB shared mem -> Use VT=4, Block=128 (Lower register pressure)
        let (vt, block_size) = if max_shared_mem >= 32768 {
            (8, 256)
        } else {
            log::warn!("Low-end GPU detected. Downgrading to VT=4.");
            (4, 128)
        };

        let config = common::shader::ShaderConfig { vt, block_size };

        let scan_pipeline = common::shader::create_compute_pipeline(
            &device,
            &bind_group_layout,
            include_str!("scan.wgsl"),
            &format!("sort/scanner:vt{vt}/pipeline:scan"),
            "main",
            Some(&config),
        );

        let add_pipeline = common::shader::create_compute_pipeline(
            &device,
            &bind_group_layout,
            include_str!("add.wgsl"),
            &format!("sort/scanner:vt{vt}/pipeline:add"),
            "main",
            Some(&config),
        );

        Self {
            bind_group_layout,
            scan_pipeline,
            add_pipeline,
            vt,
            block_size,
        }
    }

    pub fn get_scratch_size(&self, num_items: u32) -> u64 {
        let mut size = 0;
        let mut current_items = num_items;

        let items_per_block = self.vt * self.block_size;

        while current_items > 1 {
            let aux_count = (current_items + items_per_block - 1) / items_per_block;
            let raw_size = (aux_count * 4) as u64;
            let aligned_size = common::math::align_to(raw_size, 256);
            size += aligned_size;
            current_items = aux_count;
        }
        size
    }
}
