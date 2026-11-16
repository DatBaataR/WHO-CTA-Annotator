using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;

[assembly: AssemblyTitle("WHO Epidemiological Table Annotator Launcher")]
[assembly: AssemblyDescription("One-click launcher for the WHO CTA & Symptom Extraction Web GUI")]
[assembly: AssemblyCompany("WHO Thesis Project")]
[assembly: AssemblyProduct("WHO-CTA-Annotator")]
[assembly: AssemblyVersion("1.0.0.0")]

namespace WHOCTAAnnotator
{
    class Program
    {
        static bool HasRequiredPackages(string pythonExe)
        {
            try
            {
                ProcessStartInfo psi = new ProcessStartInfo
                {
                    FileName = pythonExe,
                    Arguments = "-c \"import pandas, sklearn, joblib; print('OK')\"",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                };
                using (Process p = Process.Start(psi))
                {
                    string output = p.StandardOutput.ReadToEnd();
                    p.WaitForExit(8000);
                    return p.ExitCode == 0 && output.Contains("OK");
                }
            }
            catch
            {
                return false;
            }
        }

        static bool IsValidPython(string pythonExe)
        {
            try
            {
                ProcessStartInfo psi = new ProcessStartInfo
                {
                    FileName = pythonExe,
                    Arguments = "--version",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                };
                using (Process p = Process.Start(psi))
                {
                    p.WaitForExit(5000);
                    return p.ExitCode == 0;
                }
            }
            catch
            {
                return false;
            }
        }

        static string FindBestPython(string baseDir)
        {
            List<string> candidates = new List<string>();

            // 1. Explicit environment variable override
            string envPy = Environment.GetEnvironmentVariable("WHO_CTA_PYTHON");
            if (!string.IsNullOrEmpty(envPy))
                candidates.Add(envPy);

            // 2. Local project virtual environments (.venv, venv, env)
            candidates.Add(Path.Combine(baseDir, ".venv", "Scripts", "python.exe"));
            candidates.Add(Path.Combine(baseDir, "venv", "Scripts", "python.exe"));
            candidates.Add(Path.Combine(baseDir, "env", "Scripts", "python.exe"));

            // 3. User Conda / Miniconda / Anaconda environments (prioritize 'RECA' or 'who-cta')
            string userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            string localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            string programData = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData);

            string[] condaRoots = new string[]
            {
                Path.Combine(userProfile, "anaconda3"),
                Path.Combine(userProfile, "miniconda3"),
                Path.Combine(userProfile, "miniforge3"),
                Path.Combine(localAppData, "anaconda3"),
                Path.Combine(localAppData, "miniconda3"),
                Path.Combine(localAppData, "Continuum", "anaconda3"),
                Path.Combine(programData, "anaconda3"),
                Path.Combine(programData, "miniconda3"),
                @"C:\anaconda3",
                @"C:\miniconda3"
            };

            foreach (string root in condaRoots)
            {
                candidates.Add(Path.Combine(root, "envs", "RECA", "python.exe"));
                candidates.Add(Path.Combine(root, "envs", "who-cta", "python.exe"));
                candidates.Add(Path.Combine(root, "envs", "cta", "python.exe"));

                string envsDir = Path.Combine(root, "envs");
                if (Directory.Exists(envsDir))
                {
                    try
                    {
                        foreach (string subEnv in Directory.GetDirectories(envsDir))
                        {
                            string py = Path.Combine(subEnv, "python.exe");
                            if (!candidates.Contains(py))
                                candidates.Add(py);
                        }
                    }
                    catch { }
                }
                candidates.Add(Path.Combine(root, "python.exe"));
            }

            // First pass: check existing file paths that already have pandas + scikit-learn installed
            foreach (string cand in candidates)
            {
                if (File.Exists(cand) && HasRequiredPackages(cand))
                {
                    return cand;
                }
            }

            // Second pass: check PATH 'python' if it has required packages
            if (HasRequiredPackages("python"))
            {
                return "python";
            }

            // Third pass: return any valid Python interpreter so we can offer to install requirements
            foreach (string cand in candidates)
            {
                if (File.Exists(cand) && IsValidPython(cand))
                {
                    return cand;
                }
            }

            if (IsValidPython("python"))
            {
                return "python";
            }

            return null;
        }

        static int Main(string[] args)
        {
            Console.Title = "WHO Epidemiological Table Annotator — Web GUI";
            string exePath = Assembly.GetExecutingAssembly().Location;
            string baseDir = Path.GetDirectoryName(exePath);
            string appScript = Path.Combine(baseDir, "app.py");

            if (!File.Exists(appScript))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("[ERROR] Could not find app.py in: " + baseDir);
                Console.ResetColor();
                Console.WriteLine("Please ensure WHO-CTA-Annotator.exe stays in the project root folder.");
                Console.WriteLine("\nPress any key to exit...");
                Console.ReadKey();
                return 1;
            }

            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("===========================================================================");
            Console.WriteLine("  WHO Epidemiological Column Type Annotator & Symptom Extraction Launcher");
            Console.WriteLine("===========================================================================");
            Console.ResetColor();
            Console.WriteLine("  Locating Python environment...");

            string pythonExe = FindBestPython(baseDir);
            if (pythonExe == null)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("\n[ERROR] No Python installation was found on this system.");
                Console.ResetColor();
                Console.WriteLine("Please install Python 3.7+ or Anaconda/Miniconda, then run:");
                Console.WriteLine("  pip install -r requirements.txt");
                Console.WriteLine("\nPress any key to exit...");
                Console.ReadKey();
                return 1;
            }

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("  Using Python: " + pythonExe);
            Console.ResetColor();

            if (!HasRequiredPackages(pythonExe))
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine("\n[NOTICE] Core Python packages (pandas, scikit-learn) were not found in this environment.");
                Console.ResetColor();
                Console.Write("Would you like to automatically install requirements.txt now? [Y/n]: ");
                string ans = Console.ReadLine();
                if (string.IsNullOrEmpty(ans) || ans.Trim().ToLower().StartsWith("y"))
                {
                    ProcessStartInfo pipPsi = new ProcessStartInfo
                    {
                        FileName = pythonExe,
                        Arguments = "-m pip install -r \"" + Path.Combine(baseDir, "requirements.txt") + "\"",
                        WorkingDirectory = baseDir,
                        UseShellExecute = false
                    };
                    using (Process pipProc = Process.Start(pipPsi))
                    {
                        pipProc.WaitForExit();
                    }
                }
            }

            string forwardedArgs = string.Join(" ", args);
            ProcessStartInfo runPsi = new ProcessStartInfo
            {
                FileName = pythonExe,
                Arguments = "\"" + appScript + "\" " + forwardedArgs,
                WorkingDirectory = baseDir,
                UseShellExecute = false
            };

            try
            {
                using (Process proc = Process.Start(runPsi))
                {
                    proc.WaitForExit();
                    if (proc.ExitCode != 0)
                    {
                        Console.ForegroundColor = ConsoleColor.Red;
                        Console.WriteLine("\nServer exited with code " + proc.ExitCode + ".");
                        Console.ResetColor();
                        Console.WriteLine("Press any key to close this window...");
                        Console.ReadKey();
                    }
                    return proc.ExitCode;
                }
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("\n[ERROR] Failed to launch Web GUI: " + ex.Message);
                Console.ResetColor();
                Console.WriteLine("Press any key to close...");
                Console.ReadKey();
                return 1;
            }
        }
    }
}
