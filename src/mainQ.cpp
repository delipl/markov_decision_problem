#include "DataLoader.hpp"
#include "World.hpp"
#include <iostream>
#include <limits>

struct StateData {
  int x;
  int y;
  std::vector<double> utilities;
};

int main(int argc, char *argv[]) {

  if (argc == 1) {
    std::cerr << "Add argument data path!" << std::endl;
    return -1;
  }
  DataLoader dataLoader;
  try {
    dataLoader.load(argv[1]); // Replace with your actual data file path
    dataLoader.printData();
  } catch (const std::runtime_error &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  std::srand(std::time(nullptr)); 
  const auto probabilities = dataLoader.getProbabilities();
  const auto gamma = dataLoader.getGamma();
  const auto epsilon = dataLoader.getEpsilon();

  World world(dataLoader);

  world.printWorld();

  for (int i = 0; i < 1; ++i) {
    world.QLearning();

    world.printWorld();

  }
  world.printWorld();

  return 0;
}
