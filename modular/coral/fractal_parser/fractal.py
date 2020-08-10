
import turtle
from typing import Optional

from coral.environment.descriptor_node import DescriptorNode, Graph
from coral.fractal_parser.fractal_to_graph import program_to_graph
from coral.fractal_parser.showrobot import move_robot


def place_turtle():
    """Place the turtle reasonably"""
    turtle.left(90)
    turtle.penup()
    turtle.setpos(0, -250)
    turtle.pendown()
    turtle.shape("turtle")


def generate_program(iteration, start_string, rule_input, rule_output):
    """Generates the program for turtle
    inspired in: https://gist.github.com/HansNewbie/9808789
    """
    result = list(start_string)
    rule_input = list(rule_input)
    rule_output = [list(rule) for rule in rule_output]
    temp = []

    for i in range(iteration):  # repeat N times
        for res in result:  # source characters
            for rule_in, rule_out in zip(rule_input, rule_output):  # rules
                if res == rule_in:  # input rule matches
                    temp.extend(rule_out)  # add out-rule
                    break
                # at the end of rules and none applied? constant
                elif rule_in == rule_input[-1]:
                    temp.append(res)
        result = temp.copy()
        temp = []
    return "".join(result)


def draw_fractal(program, front, turn):
    """Draws fractal using turtle
    inspired in: https://gist.github.com/HansNewbie/9808789
    """
    place_turtle()

    stack = []
    dirstack = []

    for x in program:
        if x == 'F':
            turtle.forward(front)
        elif x == '-':
            turtle.left(turn)
        elif x == '+':
            turtle.right(turn)
        elif x == '[':
            stack.append(turtle.pos())
            dirstack.append(turtle.heading())
        elif x == ']':
            turtle.penup()
            post = stack.pop()
            direc = dirstack.pop()
            turtle.setpos(post)
            turtle.setheading(direc)
            turtle.pendown()
        # else:
        #     print(f'X')
    turtle.hideturtle()
    turtle.done()


def add_offspring(parent: Optional[DescriptorNode], offspring: DescriptorNode, graph: Graph) -> DescriptorNode:
    if parent is not None:
        parent.add_child(offspring)
    else:
        graph.root = offspring
    return offspring


if __name__ == '__main__':
    """Tests how to draw a fractal using turtle, or convert program to Graph for the simulator"""

    FRONT = 10
    TURN = 33
    iters = 6

    a = True
    if a:
        ruleInput = ['F', 'X']
        ruleOutput = ["FF", "F-[[X]+X]+F[+FX]-X"]
        start = "X"
    else:
        ruleInput = ['F']
        ruleOutput = ['F[-F][+F]']
        start = "F"

    program = generate_program(iters, start, ruleInput, ruleOutput)
    print(f'program={program}')

    graph = program_to_graph(program)
    # show_robot(graph, blocking=True)
    # draw_fractal(program, FRONT, TURN)
    move_robot(graph)

