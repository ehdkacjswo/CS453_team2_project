def whichtriangle(x,y,z):
    if (x == y):
        if (x == z):
            if (y == z):
                print("Equilateral triangle")
            else:
                print("Isosceles triangle")
        elif (y == z):
            print("Isosceles triangle") 
    elif (x == z):
        if (y == z):
            print ("Isosceles triangle")
    else:
        print("Scalene triangle")


#[[1000, 12, 9, 140.40512895584106]] 
#[[1000, 12, 9, 85.82653522491455]]