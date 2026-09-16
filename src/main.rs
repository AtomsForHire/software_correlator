use std::{fmt::Debug, sync::Arc};

use num_complex::Complex;
use rustfft::{
    Fft, FftNum, FftPlanner,
    num_traits::{FromPrimitive, Signed},
};

fn main() {
    println!("Hello, world!");
}

/// Function to cross-correlate two signals. Assumes input slices are already the same length
fn correlate<T: Copy + FromPrimitive + Signed + FftNum + Debug>(
    fft: &Arc<dyn Fft<T>>,
    a: &mut [Complex<T>],
    b: &mut [Complex<T>],
) -> Vec<Complex<T>> {
    // First check that a and b are same lengths
    assert_eq!(a.len(), fft.len());
    assert_eq!(b.len(), fft.len());

    // Cross-correlation has the property that it is the product of the functions in Fourier space (I think?)
    // FFT will be done in place, so treat `a` as the output function finishes

    // So FFT the input signals first
    // this is done in-place
    fft.process(a);
    fft.process(b);

    // Now multiply them together to get the cross correlation
    a.iter()
        .zip(b.iter())
        .map(|(elem_a, elem_b)| elem_a * elem_b.conj())
        .collect()
}

#[cfg(test)]
mod tests {
    use num_complex::Complex64;

    use super::*;
    use std::f64::consts::PI;

    fn tone(size: usize, bin: usize, phase: f64) -> Vec<Complex<f64>> {
        (0..size)
            .map(|n| {
                let angle = 2.0 * PI * bin as f64 * n as f64 / size as f64 + phase;
                Complex::new(angle.cos(), angle.sin())
            })
            .collect()
    }

    /// Wraps a phase difference into (-pi, pi], so comparisons don't
    /// falsely fail at the +pi/-pi boundary.
    fn wrap_phase(theta: f64) -> f64 {
        let two_pi = 2.0 * PI;
        ((theta + PI).rem_euclid(two_pi)) - PI
    }

    /// Finds the bin with the largest magnitude in a cross-power spectrum.
    fn peak_bin(spectrum: &[Complex<f64>]) -> usize {
        spectrum
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.norm().partial_cmp(&b.norm()).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }

    #[test]
    fn identical_signals_correlate_with_zero_phase() {
        let size = 64;
        let mut a = tone(size, 5, 0.0);
        let mut b = a.clone();

        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(size);

        let result = correlate(&fft, &mut a, &mut b);
        let peak = peak_bin(&result);

        assert_eq!(peak, 5);
        assert!(wrap_phase(result[peak].arg()).abs() < 1e-9);
    }
}
