use wgpu::{Instance, MemoryHints, RequestAdapterOptions};
use wgpu_sort::sort::Sorter;

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    let instance = Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::None,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .unwrap();

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("Context Device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            memory_hints: MemoryHints::Performance,
            ..Default::default()
        })
        .await
        .unwrap();

    println!("Initializing Context...");
    let sorter = Sorter::new(&device, 12);

    let input = vec![10, 5, 8, 1, 2, 9, 3, 4, 7, 6, 0, 11];
    let keys = (0..input.len() as u32).collect::<Vec<_>>();
    println!("Input:  {:?}", input);

    let (res, _res_keys) = sorter.sort_array(&device, &queue, &input, &keys).await;
    println!("Output: {:?}", res);

    let mut expected = input.clone();
    expected.sort();

    let result_slice = &res[0..input.len()];
    assert_eq!(result_slice, expected.as_slice());
    println!("Radix Sort Verified!");
}
