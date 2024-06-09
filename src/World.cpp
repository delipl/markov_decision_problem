#include "World.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
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
  epsilon = dataLoader.getEpsilon();
  initializeGrid();

  auto probs = dataLoader.getProbabilities();

  probabilities[1] = std::get<0>(probs);
  probabilities[0] = std::get<1>(probs);
  probabilities[2] = std::get<2>(probs);

  std::cout << probabilities[0] << " " << probabilities[1] << "0.1"
            << probabilities[2] << std::endl;
}

void World::initializeGrid() {
  grid.resize(height,
              std::vector<State>(
                  width, {
                             0.0f,
                             ' ',
                             reward,
                             ' ',
                             {{'^', 0}, {'v', 0}, {'<', 0}, {'>', 0}},
                             {{'^', 0.0}, {'v', 0.0}, {'<', 0.0}, {'>', 0.0}},
                         }));

  for (const auto &ts : terminalStates) {
    grid[ts.y - 1][ts.x - 1] = {ts.reward, 'T', ts.reward, ' '};
  }

  for (const auto &ss : specialStates) {
    grid[ss.y - 1][ss.x - 1] = {
        ss.reward,
        '*',
        ss.reward,
        ' ',
        {{'^', 0}, {'v', 0}, {'<', 0}, {'>', 0}},
        {{'^', 0.0},
         {'v', 0.0},
         {'<', 0.0},
         {'>', 0.0}}}; // Special states can have specific values
  }

  for (const auto &fs : forbiddenStates) {
    grid[fs.y - 1][fs.x - 1] = {0.0f, 'F', 0.0, ' '};
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

      std::cout << grid[y][x].type;
    }
    std::cout << "|" << std::endl;

    for (int x = 0; x < width; ++x) {
      std::cout << "  |" << std::setw(10) << " ";
    }
    std::cout << "|" << std::endl;

    for (int x = 0; x < width; ++x) {
      std::cout << "  | " << grid[y][x].policy << std::setw(8) << std::fixed
                << std::setprecision(4) << grid[y][x].utility;
    }
    std::cout << "|" << std::endl;

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

void World::updateUtility(int x, int y, float utility) {
  if (x >= 1 && x <= width && y >= 1 && y <= height) {
    grid[y - 1][x - 1].utility = utility;
  } else {
    throw std::out_of_range("Coordinates out of range");
  }
}

void World::updateType(int x, int y, char policy) {
  if (x >= 1 && x <= width && y >= 1 && y <= height) {
    grid[y - 1][x - 1].type = policy;
  } else {
    throw std::out_of_range("Coordinates out of range");
  }
}

float World::getValue(int x, int y) const {
  if (x >= 1 && x <= width && y >= 1 && y <= height) {
    return grid[y - 1][x - 1].utility;
  } else {
    // throw std::out_of_range("Coordinates out of range");
    std::cerr << "Coordinates out of range" << std::endl;
    return 0.0;
  }
}

char World::getType(int x, int y) const {
  if (x >= 1 && x <= width && y >= 1 && y <= height) {
    return grid[y - 1][x - 1].type;
  } else {
    throw std::out_of_range("Coordinates out of range");
  }
}

void World::updatePolicy(int x, int y, char utility) {
  if (x >= 1 && x <= width && y >= 1 && y <= height) {
    grid[y - 1][x - 1].policy = utility;
  } else {
    throw std::out_of_range("Coordinates out of range");
  }
}

char World::getPolicy(int x, int y) const {
  if (x >= 1 && x <= width && y >= 1 && y <= height) {
    return grid[y - 1][x - 1].policy;
  } else {
    throw std::out_of_range("Coordinates out of range");
  }
}
float World::getReward(int x, int y) const {
  if (x >= 1 && x <= width && y >= 1 && y <= height) {
    return grid[y - 1][x - 1].reward;
  } else {
    throw std::out_of_range("Coordinates out of range");
  }
}

void World::addVisit(int x, int y, char action) {
  if (x >= 1 && x <= width && y >= 1 && y <= height) {
    grid[y - 1][x - 1].visits.at(action)++;
  } else {
    throw std::out_of_range("Coordinates out of range");
  }
}

uint32_t World::getVisits(int x, int y, char action) const {
  if (x >= 1 && x <= width && y >= 1 && y <= height) {
    return grid[y - 1][x - 1].visits.at(action);
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
          float oldValue = state.utility;
          float newValue = getMaxQValue(x, y);
          state.utility = newValue;
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

  std::map<char, std::vector<std::array<int, 2>>> actions = {// UP
                                                             {'^',
                                                              {
                                                                  {x - 1, y},
                                                                  {x, y + 1},
                                                                  {x + 1, y},
                                                              }},
                                                             // DOWN
                                                             {'v',
                                                              {
                                                                  {x - 1, y},
                                                                  {x, y - 1},
                                                                  {x + 1, y},
                                                              }},
                                                             // LEFT
                                                             {'<',
                                                              {
                                                                  {x, y - 1},
                                                                  {x - 1, y},
                                                                  {x, y + 1},
                                                              }},
                                                             // RIGHT
                                                             {
                                                                 '>',
                                                                 {
                                                                     {x, y - 1},
                                                                     {x + 1, y},
                                                                     {x, y + 1},
                                                                 },
                                                             }};

  float max_utility = std::numeric_limits<float>::lowest();
  char max_policy = 'o';
  for (const auto &[key, action] : actions) {
    float utility = 0.0;

    // const float probabilities[3] = {0.1, 0.8, 0.1};
    for (int i = 0; i < 3; ++i) {
      auto new_x = action[i][0];
      auto new_y = action[i][1];

      try {
        const auto target_policy = getType(new_x, new_y);
        if (target_policy == 'F') {
          utility += probabilities[i] * getValue(x, y);

        } else {
          utility += probabilities[i] * getValue(new_x, new_y);
        }
      } catch (const std::out_of_range &e) {
        utility += probabilities[i] * getValue(x, y);
      }
    }
    utility *= gamma;

    if (max_utility < utility) {
      max_policy = key;
      max_utility = utility;
    }

    // std::cout << "For state(" << x << "," << y << ") next state: ("
    //           << action[1][0] << "," << action[1][1]
    //           << ") is counted: " << utility << std::endl;
  }

  updatePolicy(x, y, max_policy);

  return getReward(x, y) + max_utility;
}

float World::getQValue(int x, int y, char action) {

  if (x >= 1 && x <= width && y >= 1 && y <= height) {
    return grid[y - 1][x - 1].q.at(action);
  } else {
    throw std::out_of_range("Coordinates out of range");
  }
}

void World::updateQValue(int x, int y, char action, float value) {
  if (x >= 1 && x <= width && y >= 1 && y <= height) {
    grid[y - 1][x - 1].q.at(action) = value;
  } else {
    throw std::out_of_range("Coordinates out of range");
  }
}

std::pair<int, int> World::getStart() {
  for (int i = 1; i <= width; ++i) {
    for (int j = 1; j <= height; ++j) {
      if (getType(i, j) == 'S') {
        return {i, j};
      }
    }
  }
  return {-1, -1};
}

void World::QLearning(int start_x, int start_y, int &x, int& y ) {

  while (getType(x, y) != 'T') {
    auto action = getRandomAction(x, y);
    // std::cout << "at (" << x << "," << y << ") " << std::endl;
    auto [new_x, new_y] = execute_action(x, y, action.first);
    // std::cout << "looking at (" << new_x << "," << new_y << ")" <<
    // std::endl;

    addVisit(x, y, action.first);
    float alpha = 1.0 / (getVisits(x, y, action.first));
    float old_q = getQValue(x, y, action.first);

    float q_max = 0.0;

    if (getType(new_x, new_y) != 'T') {
      q_max = getMaxQValue(new_x, new_y);
    } else {
      q_max = getReward(new_x, new_y);
    }

    float new_q = getReward(x, y) + gamma * q_max;
    float updated_q = old_q + alpha * (new_q - old_q);

    updateQValue(x, y, action.first, updated_q);
    q_max = getMaxQValue(x, y);
    updateUtility(x, y, q_max);

    x = new_x;
    y = new_y;
  }
}

std::pair<char, std::vector<std::array<int, 2>>> World::getRandomAction(int x,
                                                                        int y) {
  std::map<char, std::vector<std::array<int, 2>>> actions = {// DOWN
                                                             {'v',
                                                              {
                                                                  {x - 1, y},
                                                                  {x, y - 1},
                                                                  {x + 1, y},
                                                              }},
                                                             // UP
                                                             {'^',
                                                              {
                                                                  {x - 1, y},
                                                                  {x, y + 1},
                                                                  {x + 1, y},
                                                              }},
                                                             // LEFT
                                                             {'<',
                                                              {
                                                                  {x, y - 1},
                                                                  {x - 1, y},
                                                                  {x, y + 1},
                                                              }},
                                                             // RIGHT
                                                             {
                                                                 '>',
                                                                 {
                                                                     {x, y - 1},
                                                                     {x + 1, y},
                                                                     {x, y + 1},
                                                                 },
                                                             }};

  auto it = actions.begin();

  std::mt19937_64 rng;
  uint64_t timeSeed =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed >> 32)};
  rng.seed(ss);
  std::uniform_real_distribution<double> unif(0, 1);
  auto random = unif(rng);
  auto policy = getPolicy(x, y);
  std::cout << "epsilon: " << epsilon << " random: " << random << std::endl;
  if (random < epsilon or policy == ' ') {
    random = unif(rng);
    if (random < 0.75) {
      ++it;
    }
    if (random < 0.5) {
      ++it;
    }
    if (random < 0.25) {
      ++it;
    }
    return *it;
  }
  return {policy, actions[policy]};
}

std::pair<int, int> World::execute_action(int start_x, int start_y,
                                          char action) {
  auto x = start_x;
  auto y = start_y;
  if (action == '^') {
    y += 1;
  } else if (action == 'v') {
    y -= 1;
  } else if (action == '<') {
    x -= 1;
  } else if (action == '>') {
    x += 1;
  }
  try {
    if (getType(x, y) == 'F') {
      return {start_x, start_y};
    } else if (x >= 1 && x <= width && y >= 1 && y <= height) {
      return {x, y};
    } else {

      return {start_x, start_y};
    }
  } catch (const std::out_of_range &e) {
    return {start_x, start_y};
  }
}
