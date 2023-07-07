mod network;
use crate::network::network::*;
mod mnist_loader;
use crate::mnist_loader::mnist_loader::get_mnist;
mod functions;
use functions::functions::*;
use std::fs;
use png;
use std::fs::File;

fn main() {
  let train = false;
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

    let decoder = png::Decoder::new(File::open("input/number.png").unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let bytes = &buf[..info.buffer_size()].iter().map(|x| 1.0 - *x as f64/255.0).collect::<Vec<f64>>();
    let mut input = vec![0.0; 28*4];
    for i in 0..20 {
      input.append(&mut vec![0.0; 4]);
      for j in 0..20 {
        input.push(bytes[(i*20+j)*3]);
      }
      input.append(&mut vec![0.0; 4]);
    }
    input.append(&mut vec![0.0; 28*4]);
    let probabilities = network.run(&input);
    let mut guess = 0;
    let mut highest = 0.0;
    for i in 0..probabilities.len() {
      if probabilities[i] > highest {
        highest = probabilities[i];
        guess = i;
      }
    }
    println!("Probabilities: {:?} \n Guess: {}", probabilities, guess);
  }
}