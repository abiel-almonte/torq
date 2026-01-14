import dis

class Tracer:
    def __init__(self) -> None:
        self.output = []
        self.depth = 0

    def get_line(self, lasti, code):
       intrs = list(dis.get_instructions(code))
       curr_intr = next(i for i in intrs if i.offset == lasti)
       return f"{curr_intr.opname} {curr_intr.argval}"


    def __call__(self, frame, event, arg):
        frame.f_trace_opcodes = True

        if event == "call":
            indent = "  " * self.depth
            self.output.append(f"{indent}[CALL] {frame.f_code.co_name} (Line {frame.f_lineno})")
            self.depth += 1

        elif event == "line" or event == "opcode":
            indent = "  " * self.depth
            #ine_text = linecache.getline(frame.f_code.co_filename, frame.f_lineno).strip()
            line = self.get_line(frame.f_lasti, frame.f_code)
            self.output.append(f"{indent} LINE {frame.f_lineno}: {line}")

        elif event == "return":
            self.depth -= 1
            indent = "  " * self.depth
            self.output.append(f"{indent}[RETURN] {frame.f_code.co_name}")

        return self

    def print_trace(self):
        for line in self.output:
            print(line)
