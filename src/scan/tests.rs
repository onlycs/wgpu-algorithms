#[cfg(test)]
mod tests {
    use crate::common;
    use crate::context::Context;
    use crate::scan::Scanner;

    #[tokio::test]
    async fn test_scan() {
        let ctx = Context::init().await.unwrap();

        let n: u32 = 1_000_000;
        let scanner = Scanner::new(&ctx.device, n as u64 * 4);

        let input: Vec<u32> = (0..n).map(|_| rand::random::<u32>() % 100).collect();

        let cpu_result: Vec<u32> = input
            .iter()
            .scan(0u32, |state, &x| {
                *state += x;
                Some(*state)
            })
            .collect();

        ctx.queue
            .write_buffer(scanner.buf_hist(), 0, bytemuck::cast_slice(&input));

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        scanner.record_scan(&mut encoder);
        ctx.queue.submit(Some(encoder.finish()));

        let gpu_result = common::buffers::download_buffer(
            &ctx.device,
            &ctx.queue,
            scanner.buf_hist(),
            n as u64 * 4,
        )
        .await;

        assert_eq!(cpu_result, gpu_result, "GPU Scan result matches CPU");
    }
}
