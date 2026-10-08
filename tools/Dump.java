import com.cburch.logisim.file.*;
import com.cburch.logisim.circuit.*;
import com.cburch.logisim.comp.*;
import java.io.File;
public class Dump {
  public static void main(String[] a) throws Exception {
    Loader l = new Loader(null);
    LogisimFile f = l.openLogisimFile(new File(a[0]));
    for (Circuit c : f.getCircuits()) {
      System.out.println("== " + c.getName());
      for (Component comp : c.getNonWires()) {
        StringBuilder sb = new StringBuilder(comp.getFactory().getName() + " @" + comp.getLocation() + ":");
        for (EndData e : comp.getEnds()) sb.append(" " + e.getLocation() + "/w" + e.getWidth().getWidth() + (e.isInput()&&e.isOutput()?"io":e.isInput()?"i":"o"));
        System.out.println(sb);
      }
    }
    System.exit(0);
  }
}
