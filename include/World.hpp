#ifndef WORLD_HPP
#define WORLD_HPP

#include "DataLoader.hpp"
#include "Policy.hpp"
#include <iomanip> // for std::setw
#include <vector>

struct State {
  float value;
  char type;
  float reward;
  char policy;
};

class World {
public:
  World(const DataLoader &dataLoader);

  void printWorld() const;
  void updateValue(int x, int y,
                   float value); // Update the value of a specific state
  void updateType(int x, int y,
                  char policy);       // Update the policy of a specific state
  float getValue(int x, int y) const; // Get the value of a specific state
  char getType(int x, int y) const;   // Get the policy of a specific state
  int getWidth() const { return width; }
  int getHeight() const { return width; }

  void valueIteration(float gamma, float epsilon);
  float getMaxQValue(int x, int y);
  void updatePolicy(int x, int y,
                    char value); // Update the value of a specific state
  char getPolicy(int x, int y) const;
  float getReward(int x, int y) const;

private:
  int width;
  int height;
  std::vector<std::vector<State>> grid;
  std::vector<TerminalState> terminalStates;
  std::vector<SpecialState> specialStates;
  std::vector<ForbiddenState> forbiddenStates;
  std::pair<int, int> startState;
  bool startStateSet;
  float reward;
  float gamma;
  float probabilities[3]; void initializeGrid();
};

#endif // WORLD_HPP
