use crate::common;

pub struct SortPipeline {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub reduce_pipeline: wgpu::ComputePipeline,
    pub scatter_pipeline: wgpu::ComputePipeline,
    pub vt: u32,
    pub block_size: u32,
}

impl SortPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sort/sorter/bindgroup_layout"),
            entries: &[
                common::buffers::bind_entry(0, true, false), // values input
                common::buffers::bind_entry(1, false, false), // histograms
                common::buffers::bind_entry(2, false, false), // values output
                common::buffers::bind_entry(3, false, true), // uniforms
                common::buffers::bind_entry(4, true, false), // keys input
                common::buffers::bind_entry(5, false, false), // keys output
                common::buffers::bind_entry(6, true, false), // scan_aux (read-only storage)
            ],
        });

        let limits = device.limits();
        let max_shared_mem = limits.max_compute_workgroup_storage_size;

        let (vt, block_size) = if max_shared_mem >= 32768 {
            (8, 256) // M3 / Desktop
        } else {
            (4, 128) // Mobile
        };

        let config = common::shader::ShaderConfig { vt, block_size };

        common::shader::create_compute_pipeline(
            &device,
            &bind_group_layout,
            include_str!("sort.wgsl"),
            &format!("sort/sorter:vt{vt}/pipeline:reduce"),
            "main_reduce",
            Some(&config),
        );

        common::shader::create_compute_pipeline(
            &device,
            &bind_group_layout,
            include_str!("sort.wgsl"),
            &format!("sort/sorter:vt{vt}/pipeline:scatter"),
            "main_scatter",
            Some(&config),
        );

        let raw_shader = include_str!("sort.wgsl");
        let final_source = raw_shader
            .replace("{{VT}}", &vt.to_string())
            .replace("{{BLOCK_SIZE}}", &block_size.to_string());

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("sort/sorter/shader:final")),
            source: wgpu::ShaderSource::Wgsl(final_source.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sort/sorter/pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let reduce_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sort/sorter/pipeline:reduce"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main_reduce"),
            compilation_options: Default::default(),
            cache: None,
        });

        let scatter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sort/sorter/pipeline:scatter"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main_scatter"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            bind_group_layout,
            reduce_pipeline,
            scatter_pipeline,
            vt,
            block_size,
        }
    }
}
