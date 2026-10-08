import com.cburch.logisim.file.*;
import com.cburch.logisim.circuit.*;
import com.cburch.logisim.comp.*;
import java.io.File;
import java.util.*;
// prints instance port offsets (relative to loc) for the probe circuit as JSON: name -> {in:[[dx,dy,w]], out:[...]}
public class Inst {
  public static void main(String[] a) throws Exception {
    LogisimFile f = new Loader(null).openLogisimFile(new File(a[0]));
    StringBuilder sb = new StringBuilder("{");
    for (Circuit c : f.getCircuits()) {
      if (!c.getName().equals("probe")) continue;
      boolean first = true;
      for (Component comp : c.getNonWires()) {
        List<int[]> ins = new ArrayList<>(), outs = new ArrayList<>();
        for (EndData e : comp.getEnds()) {
          int[] v = {e.getLocation().getX() - comp.getLocation().getX(), e.getLocation().getY() - comp.getLocation().getY(), e.getWidth().getWidth()};
          (e.isOutput() ? outs : ins).add(v);
        }
        Comparator<int[]> cmp = (p, q) -> p[0] != q[0] ? p[0] - q[0] : p[1] - q[1];
        ins.sort(cmp); outs.sort(cmp);
        sb.append(first ? "" : ",").append("\"" + comp.getFactory().getName() + "\":{\"in\":" + j(ins) + ",\"out\":" + j(outs) + "}");
        first = false;
      }
    }
    System.out.println(sb.append("}"));
    System.exit(0);
  }
  static String j(List<int[]> l) { StringBuilder s = new StringBuilder("["); for (int i = 0; i < l.size(); i++) { int[] v = l.get(i); s.append(i>0?",":"").append("[" + v[0] + "," + v[1] + "," + v[2] + "]"); } return s.append("]").toString(); }
}
