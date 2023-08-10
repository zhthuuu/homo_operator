function [circles, r, N_circle] = read_circles(file)
    fid = fopen(file, 'r');
    line = 0;
    lattice = zeros(2);
    circles = [];
    r = 0;
    N_circle = 0;
    while true
        line = line + 1;
        string = fgetl(fid);
        if string == -1
            break;
        end
        if line == 2
            N_circle = str2double(string);
            circles = zeros(N_circle, 2);
        end
        if line == 3
            r = str2double(string);
        end
        if line == 4
            lattice(1,:) = str2num(string);
        end
        if line == 5
            lattice(2,:) = str2num(string);
        end
        if line > 5
            circles(line-5, :) = str2num(string);
        end
    end
    circles = circles / lattice;
end