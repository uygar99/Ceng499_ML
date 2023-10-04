cmake_minimum_required(VERSION 3.24)
project(code C)

set(CMAKE_C_STANDARD 11)

add_executable(code
        client.c
        server.c)
