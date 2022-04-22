#pragma once

#define DEBUG false

#define debug_printf(fmt, ...) do { if (DEBUG) printf(fmt, __VA_ARGS__); } while (0)
