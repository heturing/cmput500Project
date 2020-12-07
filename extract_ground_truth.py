from argparse import ArgumentParser
import subprocess
import os
from glob import glob
from tqdm import tqdm



def process_args():
    # All arguments are processed in this function.
    parsar = ArgumentParser()
    parsar.add_argument("-pp", "--projectspath", dest="projects_path", help="This is the path to all your projects, If several projects exists, this path should be a folder that contains each projects as a single folder.", required=True)
    parsar.add_argument("-o", "--output", dest="output", help="Specify the output file", required=False, default="output.txt")
    args = parsar.parse_args()
    return args

def get_all_projects(path):
    return glob(path + "/*/")

def construct_package_name(pp, fp):
    # Construct the package name based on the path of the project and the path of the class file
    tmp = fp[len(pp):]
    res = ".".join(tmp.split("/")[:-1])
    return res

def extract_method_name(lst):
    res = []
    for l in lst:
        if "(" in l and ")" in l:
            res.append(l.split("(")[0].split(" ")[-1])
    return res

def get_full_class_name(s):
    return s.split(" ")[-2]

def get_ground_truth_for_a_project(proj, output):
    print("Processing project: " + proj)

    # Find all .class files in this project
    command = ["find", proj, "-name", "'*.class'"]
    all_class_path = os.popen(" ".join(command)).read().split("\n")[:-1]

    # Java commmand that run soot.
    java_commmand = "/usr/lib/jvm/java-1.8.0-openjdk-amd64/bin/java -Dfile.encoding=UTF-8 -classpath /usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/charsets.jar:/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/ext/cldrdata.jar:/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/ext/dnsns.jar:/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/ext/icedtea-sound.jar:/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/ext/jaccess.jar:/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/ext/java-atk-wrapper.jar:/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/ext/localedata.jar:/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/ext/nashorn.jar:/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/ext/sunec.jar:/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/ext/sunjce_provider.jar:/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/ext/sunpkcs11.jar:/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/ext/zipfs.jar:/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/jce.jar:/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/jfr.jar:/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/jsse.jar:/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/management-agent.jar:/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/resources.jar:/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/rt.jar:/home/jiaqi/IdeaProjects/HelloWorld_Maven/target/classes:/home/jiaqi/.m2/repository/org/soot-oss/soot/4.2.1/soot-4.2.1.jar:/home/jiaqi/.m2/repository/commons-io/commons-io/2.6/commons-io-2.6.jar:/home/jiaqi/.m2/repository/org/smali/dexlib2/2.4.0/dexlib2-2.4.0.jar:/home/jiaqi/.m2/repository/com/google/code/findbugs/jsr305/1.3.9/jsr305-1.3.9.jar:/home/jiaqi/.m2/repository/com/google/guava/guava/27.1-android/guava-27.1-android.jar:/home/jiaqi/.m2/repository/com/google/guava/failureaccess/1.0.1/failureaccess-1.0.1.jar:/home/jiaqi/.m2/repository/com/google/guava/listenablefuture/9999.0-empty-to-avoid-conflict-with-guava/listenablefuture-9999.0-empty-to-avoid-conflict-with-guava.jar:/home/jiaqi/.m2/repository/org/checkerframework/checker-compat-qual/2.5.2/checker-compat-qual-2.5.2.jar:/home/jiaqi/.m2/repository/com/google/errorprone/error_prone_annotations/2.2.0/error_prone_annotations-2.2.0.jar:/home/jiaqi/.m2/repository/com/google/j2objc/j2objc-annotations/1.1/j2objc-annotations-1.1.jar:/home/jiaqi/.m2/repository/org/codehaus/mojo/animal-sniffer-annotations/1.17/animal-sniffer-annotations-1.17.jar:/home/jiaqi/.m2/repository/org/ow2/asm/asm/8.0.1/asm-8.0.1.jar:/home/jiaqi/.m2/repository/org/ow2/asm/asm-tree/8.0.1/asm-tree-8.0.1.jar:/home/jiaqi/.m2/repository/org/ow2/asm/asm-util/8.0.1/asm-util-8.0.1.jar:/home/jiaqi/.m2/repository/org/ow2/asm/asm-analysis/8.0.1/asm-analysis-8.0.1.jar:/home/jiaqi/.m2/repository/org/ow2/asm/asm-commons/8.0.1/asm-commons-8.0.1.jar:/home/jiaqi/.m2/repository/xmlpull/xmlpull/1.1.3.4d_b4_min/xmlpull-1.1.3.4d_b4_min.jar:/home/jiaqi/.m2/repository/de/upb/cs/swt/axml/2.0.0/axml-2.0.0.jar:/home/jiaqi/.m2/repository/ca/mcgill/sable/polyglot/2006/polyglot-2006.jar:/home/jiaqi/.m2/repository/de/upb/cs/swt/heros/1.2.2/heros-1.2.2.jar:/home/jiaqi/.m2/repository/org/functionaljava/functionaljava/4.2/functionaljava-4.2.jar:/home/jiaqi/.m2/repository/ca/mcgill/sable/jasmin/3.0.2/jasmin-3.0.2.jar:/home/jiaqi/.m2/repository/ca/mcgill/sable/java_cup/0.9.2/java_cup-0.9.2.jar:/home/jiaqi/.m2/repository/javax/annotation/javax.annotation-api/1.3.2/javax.annotation-api-1.3.2.jar:/home/jiaqi/.m2/repository/javax/xml/bind/jaxb-api/2.4.0-b180725.0427/jaxb-api-2.4.0-b180725.0427.jar:/home/jiaqi/.m2/repository/javax/activation/javax.activation-api/1.2.0/javax.activation-api-1.2.0.jar:/home/jiaqi/.m2/repository/org/glassfish/jaxb/jaxb-runtime/2.4.0-b180830.0438/jaxb-runtime-2.4.0-b180830.0438.jar:/home/jiaqi/.m2/repository/org/glassfish/jaxb/txw2/2.4.0-b180830.0438/txw2-2.4.0-b180830.0438.jar:/home/jiaqi/.m2/repository/com/sun/istack/istack-commons-runtime/3.0.7/istack-commons-runtime-3.0.7.jar:/home/jiaqi/.m2/repository/org/jvnet/staxex/stax-ex/1.8/stax-ex-1.8.jar:/home/jiaqi/.m2/repository/com/sun/xml/fastinfoset/FastInfoset/1.2.15/FastInfoset-1.2.15.jar:/home/jiaqi/.m2/repository/org/slf4j/slf4j-api/1.7.5/slf4j-api-1.7.5.jar:/home/jiaqi/.m2/repository/org/slf4j/slf4j-log4j12/1.7.5/slf4j-log4j12-1.7.5.jar:/home/jiaqi/.m2/repository/log4j/log4j/1.2.17/log4j-1.2.17.jar Hello"

    for cls in tqdm(all_class_path):
        class_name = cls.split("/")[-1].split(".")[0]
        class_path = "/".join(cls.split("/")[:-1])
        

        # Use javap get method information of a class
        clauses = os.popen("javap " + cls).read().split("\n")
        full_class_name = get_full_class_name(clauses[1])
        methods_name = extract_method_name(clauses)

        #print("methods_name: ", methods_name)

        for method in methods_name:
            print("Analyzing " + full_class_name + " and method " + method + " at " + cls + " With cp: " + proj)

            a = java_commmand.split(" ")
            a.extend([proj, full_class_name, method, " "])
            subprocess.Popen(a, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).wait()

            # with open(output, "a") as f:
            #     f.write(class_name + "\n")
            #     f.write(str(len(methods_name)) + "\n")
            #     for name in methods_name:
            #         f.write(name + "\n")

    print(all_class_path)
    #print(class_file_names)
    print("Process finish.")


def main():
    # Process all arguments
    args = process_args()

    # Get content of arguments
    projects_path = args.projects_path
    output = args.output

    for proj in get_all_projects(projects_path):
        # For each project in the projects_path, process it to get the ground truth.
        get_ground_truth_for_a_project(proj, output)


if __name__ == "__main__":
    main()