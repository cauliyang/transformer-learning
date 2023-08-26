import latexify


def solve(a, b, c):
    return (-b + (b**2 - 4 * a * c) ** 0.5) / (2 * a)


def main():
    print(latexify.get_latex(solve(a=1, b=2, c=3)))


if __name__ == "__main__":
    main()
