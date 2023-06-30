mod network;
use crate::network::network::*;
mod mnist_loader;
use crate::mnist_loader::mnist_loader::get_mnist;

fn main() {
  let data = get_mnist(20_000, 10_000);
  let mut training_data = data.0;
  let test_data = data.1;
  println!("starting training...");
  let mut network = Network::new(&[784, 30, 10], sigmoid, sigmoid_der);
  network.train(&mut training_data, &test_data, 10, 1000, 3.0, quadratic_cost_der, true);
}
