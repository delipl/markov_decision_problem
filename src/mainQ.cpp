#include "DataLoader.hpp"
#include "World.hpp"
#include "gnuplot-iostream.h"
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

  Gnuplot gp;
  std::vector<std::vector<std::vector<double>>> allValues(
      world.getWidth(), std::vector<std::vector<double>>(world.getHeight()));
  for (int i = 0; i < 1500; ++i) {
    std::cout << "========================[V(" << i+1
              << ")]========================" << std::endl;
    auto [start_x, start_y] = world.getStart();
    int x = start_x;
    int y = start_y;
    std::cout << "Iteration " << i << std::endl;
    world.QLearning(start_x, start_y, x, y);

    world.printWorld();
    for (int x = 0; x < world.getWidth(); ++x) {
      for (int y = 0; y < world.getHeight(); ++y) {
        allValues[x][y].push_back(world.getValue(x, y));
      }
    }
  }

  // Plot the collected data
  gp << "set title 'Value Evolution Over Iterations'\n";
  gp << "set xlabel 'Iteration'\n";
  gp << "set ylabel 'Value'\n";
  gp << "set key outside right top\n"; // Move legend outside the plot
  gp << "set grid\n";                  // Enable grid
  gp << "plot ";

  bool first = true;
  for (int x = 0; x < world.getWidth(); ++x) {
    for (int y = 0; y < world.getHeight(); ++y) {
      if (!first) {
        gp << ", ";
      }
      first = false;
      gp << "'-' with lines title 'Value (" << x << "," << y << ")'";
    }
  }
  gp << "\n";

  for (int x = 0; x < world.getWidth(); ++x) {
    for (int y = 0; y < world.getHeight(); ++y) {
      for (size_t iter = 0; iter < allValues[x][y].size(); ++iter) {
        gp << iter << " " << allValues[x][y][iter] << "\n";
      }
      gp << "e\n";
    }
  }

  return 0;
}
