mod network;
use crate::network::network::*;
mod mnist_loader;
use crate::mnist_loader::mnist_loader::get_mnist;
mod functions;
use functions::functions::*;
use std::fs;

fn main() {
  let train = true;
  let data = get_mnist(50_000, 10_000);
  let mut training_data = data.0;
  let test_data = data.1;
  if train {
    println!("starting training...");
    let mut network = Network::new(&[784, 100, 20, 10], &[ActivationFn::Sigmoid, ActivationFn::Sigmoid, ActivationFn::SoftMax]);
    network.train(&mut training_data, &test_data, 10, 50, 0.3, 0.002, Cost::CrossEntropy, true, true);
  } else {
    let contents = fs::read_to_string("pre_computed/net.json").expect("Unable to read file");
    let mut network = serde_json::from_str::<Network>(&contents).expect("Unable to deserialize file");
    println!("{:?}, {:?}", network.run(&test_data[0].0), test_data[0].1);
  }
}