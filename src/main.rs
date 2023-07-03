mod network;
use crate::network::network::*;
mod mnist_loader;
use crate::mnist_loader::mnist_loader::get_mnist;
mod activation_fns;
use activation_fns::activation_fns::*;

fn main() {
  let data = get_mnist(5_000, 10_000);
  let mut training_data = data.0;
  let test_data = data.1;
  println!("starting training...");
  let mut network = Network::new(&[784, 30, 10], &[sigmoid, sigmoid, sigmoid]);
  network.train(&mut training_data, &test_data, 10, 1000, 3.0, quadratic_cost_der, true);
}
