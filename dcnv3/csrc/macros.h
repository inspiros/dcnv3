#pragma once

#if defined(_WIN32) && !defined(TVDCN_BUILD_STATIC_LIBS)
#if defined(dcnv3_EXPORTS)
#define DCNV3_API __declspec(dllexport)
#else
#define DCNV3_API __declspec(dllimport)
#endif
#else
#define DCNV3_API
#endif

#if (defined __cpp_inline_variables) || __cplusplus >= 201703L
#define DCNV3_INLINE_VARIABLE inline
#else
#ifdef _MSC_VER
#define DCNV3_INLINE_VARIABLE __declspec(selectany)
#define HINT_MSVC_LINKER_INCLUDE_SYMBOL
#else
#define DCNV3_INLINE_VARIABLE __attribute__((weak))
#endif
#endif
