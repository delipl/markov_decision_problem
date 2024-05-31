#include "World.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

World::World(const DataLoader &dataLoader) {
  auto [w, h] = dataLoader.getWorldSize();
  width = w;
  height = h;
  startStateSet = dataLoader.getStartState().first != -1 &&
                  dataLoader.getStartState().second != -1;

  if (startStateSet) {
    startState = dataLoader.getStartState();
  }

  terminalStates = dataLoader.getTerminalStates();
  specialStates = dataLoader.getSpecialStates();
  forbiddenStates = dataLoader.getForbiddenStates();
  gamma = dataLoader.getGamma();
  reward = dataLoader.getDefaultReward();
  initializeGrid();
}

void World::initializeGrid() {
  grid.resize(height, std::vector<State>(width, {0.0f, ' '}));
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {

      grid[i][j].policy = CreatePoliciesForPoint(i + 1, j + 1);
    }
  }
  for (const auto &ts : terminalStates) {
    grid[ts.y - 1][ts.x - 1] = {ts.reward, 'T'};
  }

  for (const auto &ss : specialStates) {
    grid[ss.y - 1][ss.x - 1] = {ss.reward,
                                ' '}; // Special states can have specific values
  }

  for (const auto &fs : forbiddenStates) {
    grid[fs.y - 1][fs.x - 1] = {0.0f, 'F'};
  }

  if (startStateSet) {
    grid[startState.second - 1][startState.first - 1].type = 'S';
  }
}

void World::printWorld() const {
  for (int y = height - 1; y >= 0; --y) {
    for (int x = 0; x < width; ++x) {
      std::cout << "  +----------";
    }
    std::cout << "+" << std::endl;

    for (int x = 0; x < width; ++x) {
      std::cout << "  |" << std::setw(10);
      if (grid[y][x].type == 'F') {
        std::cout << "F";
      } else if (grid[y][x].type == 'T') {
        std::cout << "T";
      } else if (grid[y][x].type == 'S') {
        std::cout << "S";
      } else {
        std::cout << " ";
      }
    }
    std::cout << "  |" << std::endl;

    for (int x = 0; x < width; ++x) {
      std::cout << "  |" << std::setw(10) << " ";
    }
    std::cout << "  |" << std::endl;

    for (int x = 0; x < width; ++x) {
      std::cout << "  |" << std::setw(10) << std::fixed << std::setprecision(4)
                << grid[y][x].value;
    }
    std::cout << "  |" << std::endl;

    for (int x = 0; x < width; ++x) {
      std::cout << "  +----------";
    }
    std::cout << "+" << std::endl;
  }

  for (int x = 1; x <= width; ++x) {
    std::cout << std::setw(12) << x;
  }
  std::cout << std::endl;
}

void World::updateValue(int x, int y, float value) {
  if (x >= 1 && x <= width && y >= 1 && y <= height) {
    grid[y - 1][x - 1].value = value;
  } else {
    throw std::out_of_range("Coordinates out of range");
  }
}

void World::updatePolicy(int x, int y, char policy) {
  if (x >= 1 && x <= width && y >= 1 && y <= height) {
    grid[y - 1][x - 1].type = policy;
  } else {
    throw std::out_of_range("Coordinates out of range");
  }
}

float World::getValue(int x, int y) const {
  if (x >= 1 && x <= width && y >= 1 && y <= height) {
    return grid[y - 1][x - 1].value;
  } else {
    // throw std::out_of_range("Coordinates out of range");
    std::cerr << "Coordinates out of range" << std::endl;
    return 0.0;
  }
}

State &World::getState(int x, int y) {
  if (x >= 1 && x <= width && y >= 1 && y <= height) {
    return grid[y - 1][x - 1];
  } else {
    throw std::out_of_range("Coordinates out of range");
  }
}

char World::getType(int x, int y) const {
  if (x >= 1 && x <= width && y >= 1 && y <= height) {
    return grid[y - 1][x - 1].type;
  } else {
    throw std::out_of_range("Coordinates out of range");
  }
}

void World::valueIteration(float gamma, float epsilon) {
  this->gamma = gamma;
  bool stop_condition = false;
  float max_delta = std::numeric_limits<float>::max();
  while (!stop_condition) {
    float current_max_delta = 0.0f;

    for (int y = 1; y <= height; ++y) {
      for (int x = 1; x <= width; ++x) {
        auto &state = grid[y - 1][x - 1];
        if (state.type != 'T' && state.type != 'F') {
          float oldValue = state.value;
          float newValue = getMaxQValue(x, y);
          state.value = newValue;
          float utility_delta = std::abs(newValue - oldValue);
          if (utility_delta > epsilon) {
            max_delta = utility_delta;
          }
        }
      }
    }

    stop_condition = (current_max_delta < 0.0001f) || stop_condition;
    max_delta = (current_max_delta > max_delta) ? current_max_delta : max_delta;
  }
}

float World::getMaxQValue(int x, int y) {
  if (x < 1 || x > width || y < 1 || y > height) {
    // throw std::out_of_range("Coordinates out of range");
    std::cerr << "Coordinates out of range" << std::endl;
    return 0.0;
  }
  std::cout << "Looking at of (" << x << "," << y << ")" << std::endl;
  for (auto &[key, policies] : getState(x, y).policy) {
    const auto move = policies.first;
    auto &utility = policies.second;
    utility = 0.0;

    const float probabilities[3] = {0.1, 0.8, 0.1};

    for (int i = 0; i < 3; ++i) {
      auto new_x = move[i][0];
      auto new_y = move[i][1];

      try {
        const auto target_policy = getType(new_x, new_y);
        if (target_policy == 'F') {
          utility += probabilities[i] * getValue(x, y);

        } else {
          utility += probabilities[i] * getValue(new_x, new_y);
          std::cout << "Value of (" << new_x << "," << new_y << ") is "
                    << getValue(new_x, new_y)<< " for move at index "<< i << " in action " << key << std::endl;
        }
      } catch (const std::out_of_range &e) {
        policies.second += probabilities[i] * getValue(x, y);
      }
    }
  }

  float max_utility = std::numeric_limits<float>::lowest();
  char max_action = '^';

  for (auto &[key, policies] : getState(x, y).policy) {
    if (max_utility < policies.second) {
      max_utility = policies.second;
      max_action = key;
    }

    auto new_x = policies.first[1][0];
    auto new_y = policies.first[1][1];
    std::cout << "In Filed (" << x << "," << y << "). For action: " << key
              << " to field "
              << "(" << new_x << "," << new_y << ")"
              << " utility is: " << policies.second << std::endl;
  }

  return reward + max_utility;
}