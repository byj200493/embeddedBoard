#pragma once
#ifndef __VERSION_SOFT_HPP__
#define __VERSION_SOFT_HPP__

namespace SN
{
#define THREECAMSCAN_VERSION_MAJOR 0
#define THREECAMSCAN_VERSION_MINOR 1
#define THREECAMSCAN_VERSION_PATCH 3

#ifndef STRINGIFY
#define STRINGIFY(arg) #arg
#endif
#ifndef VAR_ARG_STRING
#define VAR_ARG_STRING(arg) STRINGIFY(arg)
#endif

// note that each component is limited into [0-99] range by design
#define THREECAMSCAN_VERSION (((THREECAMSCAN_VERSION_MAJOR) * 10000) + ((THREECAMSCAN_VERSION_MINOR) * 100) + (THREECAMSCAN_VERSION_PATCH))
#define THREECAMSCAN_VERSION_STR (VAR_ARG_STRING(THREECAMSCAN_VERSION_MAJOR.THREECAMSCAN_VERSION_MINOR.THREECAMSCAN_VERSION_PATCH))
}

#endif	// !__VERSION_HPP__
