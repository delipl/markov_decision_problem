#ifndef WORLD_HPP
#define WORLD_HPP

#include "DataLoader.hpp"
#include "Policy.hpp"
#include <iomanip> // for std::setw
#include <vector>

struct State {
  float utility;
  char type;
  float reward;
  char policy;
  std::map<char, uint32_t> visits;
  std::map<char, float> q;
  // std::pair<char, std::vector<std::array<int, 2>>> policies;
};

class World {
public:
  World(const DataLoader &dataLoader);

  void printWorld() const;
  void updateUtility(int x, int y,
                   float utility); // Update the utility of a specific state
  void updateType(int x, int y,
                  char policy);       // Update the policy of a specific state
  float getValue(int x, int y) const; // Get the utility of a specific state
  char getType(int x, int y) const;   // Get the policy of a specific state
  int getWidth() const { return width; }
  int getHeight() const { return width; }

  void valueIteration(float gamma, float epsilon);
  float getMaxQValue(int x, int y);
  void updatePolicy(int x, int y,
                    char utility); // Update the utility of a specific state
  char getPolicy(int x, int y) const;
  float getReward(int x, int y) const;
  void addVisit(int x, int y, char action);
  uint32_t getVisits(int x, int y, char action) const;
  void QLearning(int start_x, int start_y, int &x, int& y);

  float getQValue(int x, int y, char action);
  void updateQValue(int x, int y, char action, float value);

  std::pair<int, int> getStart();

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
  float epsilon;
  float probabilities[3];
  void initializeGrid();

  std::pair<char, std::vector<std::array<int, 2>>> getRandomAction(int x,
                                                                   int y);
  std::pair<int, int> execute_action(int start_x, int start_y, char action);
};

#endif // WORLD_HPP
