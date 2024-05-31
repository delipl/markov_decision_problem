#include "DataLoader.hpp"

void DataLoader::load(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        parseLine(line);
    }

    file.close();

    validateData();
}

void DataLoader::parseLine(const std::string& line) {
    std::istringstream iss(line);
    std::string label;
    iss >> label;

    if (label == "W") {
        if (!(iss >> worldWidth >> worldHeight)) {
            throw std::runtime_error("Invalid format for world size");
        }
        worldSizeSet = true;
    } else if (label == "S") {
        if (!(iss >> startX >> startY)) {
            throw std::runtime_error("Invalid format for start state");
        }
        startStateSet = true;
    } else if (label == "P") {
        if (!(iss >> p1 >> p2 >> p3)) {
            throw std::runtime_error("Invalid format for probabilities");
        }
        if (p1 < 0.0f || p2 < 0.0f || p3 < 0.0f || p1 + p2 + p3 > 1.0f) {
            throw std::runtime_error("Invalid probabilities values");
        }
        probabilitiesSet = true;
    } else if (label == "R") {
        if (!(iss >> defaultReward)) {
            throw std::runtime_error("Invalid format for default reward");
        }
        defaultRewardSet = true;
    } else if (label == "G") {
        if (!(iss >> gamma)) {
            throw std::runtime_error("Invalid format for gamma");
        }
        if (gamma <= 0.0f || gamma > 1.0f) {
            throw std::runtime_error("Invalid gamma value");
        }
        gammaSet = true;
    } else if (label == "E") {
        if (!(iss >> epsilon)) {
            throw std::runtime_error("Invalid format for epsilon");
        }
        epsilonSet = true;
    } else if (label == "T") {
        TerminalState ts;
        if (!(iss >> ts.x >> ts.y >> ts.reward)) {
            throw std::runtime_error("Invalid format for terminal state");
        }
        terminalStates.push_back(ts);
    } else if (label == "B") {
        SpecialState ss;
        if (!(iss >> ss.x >> ss.y >> ss.reward)) {
            throw std::runtime_error("Invalid format for special state");
        }
        specialStates.push_back(ss);
    } else if (label == "F") {
        ForbiddenState fs;
        if (!(iss >> fs.x >> fs.y)) {
            throw std::runtime_error("Invalid format for forbidden state");
        }
        forbiddenStates.push_back(fs);
    } else {
        throw std::runtime_error("Unknown label: " + label);
    }
}

void DataLoader::validateData() const {
    if (!worldSizeSet) {
        throw std::runtime_error("World size is not set");
    }
    if (!probabilitiesSet) {
        throw std::runtime_error("Probabilities are not set");
    }
    if (!defaultRewardSet) {
        throw std::runtime_error("Default reward is not set");
    }
    if (terminalStates.empty()) {
        throw std::runtime_error("At least one terminal state is required");
    }

    for (const auto& ts : terminalStates) {
        if (ts.x < 1 || ts.x > worldWidth || ts.y < 1 || ts.y > worldHeight) {
            throw std::runtime_error("Terminal state out of world bounds");
        }
    }

    for (const auto& ss : specialStates) {
        if (ss.x < 1 || ss.x > worldWidth || ss.y < 1 || ss.y > worldHeight) {
            throw std::runtime_error("Special state out of world bounds");
        }
    }

    for (const auto& fs : forbiddenStates) {
        if (fs.x < 1 || fs.x > worldWidth || fs.y < 1 || fs.y > worldHeight) {
            throw std::runtime_error("Forbidden state out of world bounds");
        }
    }
}

void DataLoader::printData() const {
    std::cout << "World size: " << worldWidth << " x " << worldHeight << std::endl;
    if (startStateSet) {
        std::cout << "Start state: (" << startX << ", " << startY << ")" << std::endl;
    }
    std::cout << "Probabilities: p1=" << p1 << ", p2=" << p2 << ", p3=" << p3 << std::endl;
    std::cout << "Default reward: " << defaultReward << std::endl;
    if (gammaSet) {
        std::cout << "Gamma: " << gamma << std::endl;
    }
    if (epsilonSet) {
        std::cout << "Epsilon: " << epsilon << std::endl;
    }

    std::cout << "Terminal states:" << std::endl;
    for (const auto& ts : terminalStates) {
        std::cout << "  (" << ts.x << ", " << ts.y << ") reward: " << ts.reward << std::endl;
    }

    std::cout << "Special states:" << std::endl;
    for (const auto& ss : specialStates) {
        std::cout << "  (" << ss.x << ", " << ss.y << ") reward: " << ss.reward << std::endl;
    }

    std::cout << "Forbidden states:" << std::endl;
    for (const auto& fs : forbiddenStates) {
        std::cout << "  (" << fs.x << ", " << fs.y << ")" << std::endl;
    }
}



std::pair<int, int> DataLoader::getWorldSize() const {
    return {worldWidth, worldHeight};
}

std::pair<int, int> DataLoader::getStartState() const {
    if (!startStateSet) {
        throw std::runtime_error("Start state is not set");
    }
    return {startX, startY};
}

std::tuple<float, float, float> DataLoader::getProbabilities() const {
    if (!probabilitiesSet) {
        throw std::runtime_error("Probabilities are not set");
    }
    return {p1, p2, p3};
}

float DataLoader::getDefaultReward() const {
    if (!defaultRewardSet) {
        throw std::runtime_error("Default reward is not set");
    }
    return defaultReward;
}

float DataLoader::getGamma() const {
    return gamma;
}

float DataLoader::getEpsilon() const {
    return epsilon;
}

const std::vector<TerminalState>& DataLoader::getTerminalStates() const {
    return terminalStates;
}

const std::vector<SpecialState>& DataLoader::getSpecialStates() const {
    return specialStates;
}

const std::vector<ForbiddenState>& DataLoader::getForbiddenStates() const {
    return forbiddenStates;
}