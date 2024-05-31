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
  DataLoader dataLoader;
  try {
    dataLoader.load(argv[1]); // Replace with your actual data file path
    dataLoader.printData();
  } catch (const std::runtime_error &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  const auto probabilities = dataLoader.getProbabilities();
  const auto gamma = dataLoader.getGamma();
  const auto epsilon = dataLoader.getEpsilon();

  World world(dataLoader);

  world.printWorld();

  for (int i = 0; i < 20; ++i) {
    world.valueIteration(gamma, epsilon);
  world.printWorld();

    // std::cout << "========================[V(" << i
    //           << ")]========================" << std::endl;
  }
  world.printWorld();

  //   world.valueIteration(gamma, 0.0001);

  //   std::cout << "========================[V(" << i
  //             << ")]========================" << std::endl;
  //   world.printWorld();

  return 0;
}
