#pragma once
#include <array>
#include <map>
using  PolicyMove = std::array<std::array<int, 2>, 3>;
using PolicyMoveWithReward = std::pair<PolicyMove, float>;
using Policy = std::map<char, PolicyMoveWithReward>;



Policy CreatePoliciesForPoint(int x, int y);