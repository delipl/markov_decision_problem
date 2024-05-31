#ifndef DATALOADER_HPP
#define DATALOADER_HPP

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

struct TerminalState {
  int x;
  int y;
  float reward;
};

struct SpecialState {
  int x;
  int y;
  float reward;
};

struct ForbiddenState {
  int x;
  int y;
};

class DataLoader {
public:
  void load(const std::string &filename);

  void printData() const;

  // Getters
  std::pair<int, int> getWorldSize() const;
  std::pair<int, int> getStartState() const;
  std::tuple<float, float, float> getProbabilities() const;
  float getDefaultReward() const;
  float getGamma() const;
  float getEpsilon() const;
  const std::vector<TerminalState> &getTerminalStates() const;
  const std::vector<SpecialState> &getSpecialStates() const;
  const std::vector<ForbiddenState> &getForbiddenStates() const;

private:
  void parseLine(const std::string &line);
  void validateData() const;

  int worldWidth;
  int worldHeight;
  bool worldSizeSet = false;

  int startX = -1;
  int startY = -1;
  bool startStateSet = false;

  float p1, p2, p3;
  bool probabilitiesSet = false;

  float defaultReward;
  bool defaultRewardSet = false;

  float gamma = 1.0f;
  bool gammaSet = false;

  float epsilon = 0.0f;
  bool epsilonSet = false;

  std::vector<TerminalState> terminalStates;
  std::vector<SpecialState> specialStates;
  std::vector<ForbiddenState> forbiddenStates;
};

#endif // DATALOADER_HPP
