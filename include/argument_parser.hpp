#pragma once

#include "app_config.hpp"

AppConfig parseArguments(int argc, char** argv);
void printHelp();
void printConfig(const AppConfig& config);