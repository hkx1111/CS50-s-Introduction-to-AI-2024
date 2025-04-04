import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        # raise NotImplementedError
        for var in self.crossword.variables:
            for word in self.domains[var].copy():
                if len(word) != var.length:
                    self.domains[var].remove(word)


    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        # raise NotImplementedError
        overlap =  self.crossword.overlaps.get((x, y))
        if overlap is None:
            return False
        i, j = overlap
        
        revised = False
        for word_x in self.domains[x].copy():
            delete_x_word = True
            for word_y in self.domains[y]:
                if word_x[i] == word_y[j]:
                    delete_x_word = False
                    break
                
            if delete_x_word:
                self.domains[x].remove(word_x)
                revised = True
        return revised

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        # raise NotImplementedError
        # Init queue
        queue = []
        if arcs:
            queue = arcs
        else:
            for v1 in self.crossword.variables:
                for v2 in self.crossword.neighbors(v1):
                    # if v1 != v2:
                    queue.append((v1, v2))
        
        while queue:
            arc = queue.pop(0)
            x, y = arc[0], arc[1]
            revised = self.revise(x, y)
            if revised:
                if len(self.domains[x]) == 0:
                    return False
                
                for z in self.crossword.neighbors(x):
                    if z != y:
                        queue.append((z, x))
            # x_neighbors = self.crossword.neighbors(x)
            # for z in x_neighbors:
            #     queue.append((z, x))
            
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        # raise NotImplementedError
        return len(self.crossword.variables) == len(assignment)

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        # raise NotImplementedError
        # 1. Check for uniqueness of values
        words_assigned = list(assignment.values())
        if len(words_assigned) != len(set(words_assigned)):
            return False  # If the number of unique words is not equal to the total number of words, there are duplicates

        # 2. Check if the length of each value matches the variable's length
        for var, word in assignment.items():
            if var.length != len(word):
                return False  # Word length does not match the variable's length

        # 3. Check compatibility between neighbors (overlap conflicts)
        # Iterate over every pair of variables in the assignment
        assigned_vars = list(assignment.keys())
        for i in range(len(assigned_vars)):
            for j in range(i + 1, len(assigned_vars)):  # Avoid duplicate and self-comparisons
                var1 = assigned_vars[i]
                var2 = assigned_vars[j]

                # Get the overlap information between these two variables
                # Use .get() to handle cases where the key might not exist in overlaps (though it should exist theoretically)
                overlap = self.crossword.overlaps.get((var1, var2)) 

                # If there is an overlap
                if overlap is not None:
                    idx1, idx2 = overlap  # Get the overlap indices (var1's idx1-th letter, var2's idx2-th letter)
                    word1 = assignment[var1]
                    word2 = assignment[var2]
                    
                    # Check if the letters at the overlapping positions are the same
                    if word1[idx1] != word2[idx2]:
                        return False  # Conflict exists

        # If all checks pass, the assignment is consistent
        return True
        

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        # raise NotImplementedError
        # Dictionary to store how many values each possible value rules out
        constraints = {}
        
        # Get unassigned neighbors
        neighbors = [n for n in self.crossword.neighbors(var) if n not in assignment]
        
        # For each possible value in var's domain
        for value in self.domains[var]:
            eliminated_count = 0
            
            # For each unassigned neighbor
            for neighbor in neighbors:
                # Get overlap information
                overlap = self.crossword.overlaps.get((var, neighbor))
                if overlap is None:
                    continue
                    
                var_idx, neigh_idx = overlap
                
                # Count how many values in neighbor's domain would be eliminated
                for neigh_value in self.domains[neighbor]:
                    if value[var_idx] != neigh_value[neigh_idx]:
                        eliminated_count += 1
            
            constraints[value] = eliminated_count
        
        # Return domain values sorted by how many constraints they impose (least to most)
        return sorted(self.domains[var], key=lambda val: constraints[val])
        
    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        # raise NotImplementedError
        # return [var for var in self.crossword.variables if var not in assignment][0]
        unassigned_vars = [var for var in self.crossword.variables if var not in assignment]
        
        unassigned_vars.sort(key=lambda var: len(self.domains[var]))
        
        min_domain_size = len(self.domains[unassigned_vars[0]])
        min_var = [var for var in unassigned_vars if len(self.domains[var]) == min_domain_size]
        
        if len(min_var) > 1:
            return max(min_var, key=lambda var: len(self.crossword.neighbors(var)))
        return min_var[0]

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        # raise NotImplementedError
        # 1. Base Case: Check if assignment is complete
        if self.assignment_complete(assignment):
            return assignment
        # 2. Select an unassigned variable
        var = self.select_unassigned_variable(assignment)
        # 3. Iterate through possible values for the selected variable
        for value in self.order_domain_values(var, assignment):
            # 4. Try assigning the value and check consistency
            new_assignment = assignment.copy()
            new_assignment[var] = value
            
            # 5. Check consistence
            if self.consistent(new_assignment):
                # 6. Recursive Call: If consistent, proceed recursively
                result = self.backtrack(new_assignment)
                
                if result is not None: return result
            # 7. Backtrack: If value was inconsistent or recursion failed,
            #    the loop continues to the next value. 

        return None
    
def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
