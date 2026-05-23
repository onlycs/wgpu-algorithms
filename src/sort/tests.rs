#[cfg(test)]
mod tests {
    use crate::context::Context;
    use crate::sort::Sorter;

    #[tokio::test]
    async fn test_sort() {
        let ctx = Context::init().await.unwrap();
        let n = 1_234_567u32;
        let sorter = Sorter::new(&ctx.device, n);

        let values: Vec<u32> = (0..n).map(|_| rand::random::<u32>()).collect();
        let keys: Vec<u32> = (0..n).collect();

        let mut cpu_pairs: Vec<(u32, u32)> =
            values.iter().copied().zip(keys.iter().copied()).collect();
        cpu_pairs.sort_by_key(|&(v, _)| v);
        let (cpu_values, cpu_keys): (Vec<u32>, Vec<u32>) = cpu_pairs.into_iter().unzip();

        let (gpu_values, gpu_keys) = sorter
            .sort_array(&ctx.device, &ctx.queue, &values, &keys)
            .await;

        assert_eq!(cpu_values, gpu_values, "GPU sort values match CPU");
        assert_eq!(cpu_keys, gpu_keys, "GPU sort keys match CPU");
    }
}
