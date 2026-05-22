use wgpu::{Device, Instance, MemoryHints, Queue, RequestAdapterOptions};

pub struct Context {
    pub device: Device,
    pub queue: Queue,
}

impl Context {
    pub async fn init() -> Option<Self> {
        let instance = Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::None,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok()?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Context Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                memory_hints: MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .ok()?;

        Some(Self { device, queue })
    }
}
