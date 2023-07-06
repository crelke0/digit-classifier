mod network;
use crate::network::network::*;
mod mnist_loader;
use crate::mnist_loader::mnist_loader::get_mnist;
mod functions;
use functions::functions::*;

fn main() {
  let data = get_mnist(1_000, 10_000);
  let mut training_data = data.0;
  let test_data = data.1;
  println!("starting training...");
  let mut network = Network::new(&[784, 100, 20, 10], &[ActivationFn::Sigmoid, ActivationFn::Sigmoid, ActivationFn::Sigmoid, ActivationFn::SoftMax]);
  network.train(&mut training_data, &test_data, 10, 100, 0.3, 0.002, Cost::CrossEntropy, true);
}