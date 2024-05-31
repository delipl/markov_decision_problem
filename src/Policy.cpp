#include "Policy.hpp"

Policy CreatePoliciesForPoint(int x, int y) {
  return {
        {'^', {PolicyMove{{{x - 1, y}, {x, y + 1}, {x + 1, y}}}, 0.0f}},
        {'v', {PolicyMove{{{x - 1, y}, {x, y - 1}, {x + 1, y}}}, 0.0f}},
        {'<', {PolicyMove{{{x, y - 1}, {x - 1, y}, {x, y + 1}}}, 0.0f}},
        {'>', {PolicyMove{{{x, y + 1}, {x + 1, y}, {x, y - 1}}}, 0.0f}},
  };
}
