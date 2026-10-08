import com.cburch.logisim.proj.*;
import com.cburch.logisim.gui.test.TestThread;
import java.io.File;
public class RunTV {
  public static void main(String[] a) throws Exception {
    Project p = ProjectActions.doOpenNoWindow(null, new File(a[0]));
    int r = TestThread.doTestVector(p, p.getLogisimFile().getCircuit(a[1]), a[2]);
    System.out.println("RESULT " + r);
    System.exit(r == 0 ? 0 : 1);
  }
}
