#include <vector>
#include <iostream>
#include <algorithm>
#include <string>
#include <string_view>
#include <exception>

template <typename T>
constexpr auto type_name()
{
    std::string_view name, prefix, suffix;
#ifdef __clang__
    name = __PRETTY_FUNCTION__;
    prefix = "auto type_name() [T = ";
    suffix = "]";
#elif defined(__GNUC__)
    name = __PRETTY_FUNCTION__;
    prefix = "constexpr auto type_name() [with T = ";
    suffix = "]";
#elif defined(_MSC_VER)
    name = __FUNCSIG__;
    prefix = "auto __cdecl type_name<";
    suffix = ">(void)";
#endif
    name.remove_prefix(prefix.size());
    name.remove_suffix(suffix.size());
    return name;
}

template<typename T1, typename T2>
auto add(const std::vector<T1>& a, const std::vector<T2>& b)
{
    using T = decltype(T1{} + T2{});
    std::vector<T> result;
    if (a.empty() && b.empty())
    {
        throw std::runtime_error("Empty inputs.");
    }
    for (size_t i = 0; i < std::min(a.size(), b.size()); i++)
    {
        result.emplace_back(a[i] + b[i]);
    }
    return result;
}

int main()
{
    std::wstring s = L"hello, world";
    auto c = std::ranges::count(s, L' ') + 1;
    std::cout << "there are " << c << " words in s" << std::endl;

    // Sum up two vectors.
    std::vector<int> x = { 1, 2, 3 };
    std::vector<float> y = { 3.F, 7.F, 5.F };
    auto result = add(x, y);
    for (const auto& i : result)
    {
        std::cout << type_name<decltype(i)>() << ":\t" << i << std::endl;
    }

    return 0;
}